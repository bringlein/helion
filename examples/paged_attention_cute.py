"""
Paged Attention Example
=======================

Causal attention over a paged KV cache with variable-length queries, as used
by serving engines such as vLLM: keys and values live in a pool of fixed-size
pages, each sequence addresses its pages through a block table, and a batch
of decode and prefill requests is packed into one ``[total_q_tokens, ...]``
query tensor.

Based on the PyTorch blog post "Portable paged attention in Helion"
(https://pytorch.org/blog/portable-paged-attention-in-helion/) and on the
vLLM Triton attention backend ``triton_unified_attention.py``:
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py

Structure of the kernel (GQA supported; one program handles one query head):

- **Per-sequence, per-head grid**: programs are (sequence, query head)
  pairs. Each walks its own query tokens in ``block_q`` tiles through a
  device-bounded loop and its own KV pages through scalar loops, so the
  bounds are exact: no padded ``[num_seqs, heads, max_query_len]`` grid and
  no device-side sequence resolution. The query heads of a GQA group walk
  the same pages in separate programs; those re-reads amortize through L2.
- **Affine, reshape-free tiles**: the query tile is ``[block_q, head_dim]``
  (``q[tok_tile, head, :]``) and each KV tile is one page,
  ``[page_size, head_dim]`` (``key_cache[page, :, kv_head, :]``) — grid
  tiles, scalars and slices only. The CuTe backend's SIMT/MMA lowering
  admits only such affine operand subscripts: computed gathers defeat its
  MMA operand proofs, and reshapes that merge tile dimensions move
  elements between threads, so both are refused — while Triton lowers all
  of them. That is the cost of this form versus the packed formulation
  (query rows grouped per KV head, multi-page KV tiles): it has no head
  packing in the matmul's M axis and one-page KV tiles, so a deployment
  should measure before trusting its default configs.
- **Online (flash-style) softmax** with ``exp2`` (``log2(e)`` is folded into
  the scale host-side) over one page at a time.
- **Page loop split** at the causal diagonal: pages entirely below the
  diagonal are visible to every query row of the tile, so they run without
  the causal mask and the ``seq_len`` zero-fill; only the boundary pages
  around the diagonal carry masks.
- **Ladder specialization**: ``q_block_padded_size`` is the query length
  rounded up to a power of two, so decode and prefill batches of different
  lengths autotune separately. At the same time, the ladder is required to
  not compile & autotune every different shape that happens to be
  scheduled by vLLM.
- **Default configs**: a small list of query tile widths covering the
  serving regimes. The autotuner measures them per specialization key and
  keeps the fastest, which keeps the example quick and deterministic — but
  the shapes are informed by measurements of the source kernel on sm_100
  GPUs and are not optimal on other targets or workloads. Remove
  ``configs=`` from the decorator to let the autotuner search the full
  space.

"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
import helion.language as hl

# %%
# Constants and Default Configs
# -----------------------------

# %%
# log2(e), folded into the softmax scale so the kernel can use exp2
LOG2E = 1.4426950408889634

# ceiling of the q_block_padded_size ladder, in query tokens
MAX_Q_BLOCK_PADDED = 8192


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


# Default query tile widths, informed by measurements of the source kernel:
# block_sizes are (query tokens per tile,); the KV tile is one page wide by
# construction. The autotuner measures these per specialization key and keeps
# the fastest; configs whose query tile exceeds the ladder ceiling of a key
# (the prefill shapes under a decode key) are dropped automatically.
#
# These shapes were tuned for the serving regimes above (GQA with
# num_queries_per_kv == 4, head_dim 128, page_size 16) and are not
# necessarily equally good for every batch: other head counts, head dims or
# page sizes shift the ideal query tile, so a deployment should drop this
# list (letting the autotuner search the full space) or replace it with
# shapes measured on its own distribution.
DEFAULT_CONFIGS = [
    # prefill / mixed: wide query tile
    helion.Config(block_sizes=[64], num_warps=4, num_stages=2, pid_type="flat"),
    # short prefill
    helion.Config(block_sizes=[32], num_warps=4, num_stages=2, pid_type="flat"),
    # decode: one query token per tile
    helion.Config(block_sizes=[1], num_warps=4, num_stages=2, pid_type="flat"),
]

# %%
# Paged Attention Kernel
# ----------------------


# %%
# Based on the kernel presented in the PyTorch blog post "Portable paged attention
# in Helion" (https://pytorch.org/blog/portable-paged-attention-in-helion/).
# Unlike a naive implementation, which materializes each sequence's full
# [q, kv] score matrix, this kernel runs an online softmax over one KV page
# at a time and drops masks on pages entirely below the causal diagonal.
@helion.kernel(
    configs=DEFAULT_CONFIGS,
    static_shapes=False,
    index_dtype=torch.int64,
)
def paged_attention_kernel(
    q: torch.Tensor,  # [total_q_tokens, num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,  # same layout as key_cache
    block_table: torch.Tensor,  # [num_seqs, block_table_stride]
    seq_lens: torch.Tensor,  # [num_seqs]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1] query-token offsets
    scale: float,  # sm_scale * LOG2E
    q_block_padded_size: hl.constexpr,
) -> torch.Tensor:
    """
    Computes causal attention over a paged KV cache; see the module docstring.

    Args:
        q: Packed queries, one row per token.
        key_cache: Pooled keys, one page per leading row.
        value_cache: Pooled values, same layout as ``key_cache``.
        block_table: Page ids of each sequence, zero-padded per row.
        seq_lens: Total (context + query) length of each sequence.
        cu_seqlens_q: Exclusive scan of the per-sequence query lengths.
        scale: Softmax scale pre-multiplied by ``log2(e)``.
        q_block_padded_size: Ladder bucket, ``next_power_of_2(max_query_len)``;
            the ceiling of the query tile's width.

    Returns:
        Output tensor of shape [total_q_tokens, num_heads, head_dim].
    """
    num_heads = hl.specialize(q.size(1))
    head_dim = hl.specialize(q.size(2))
    num_kv_heads = hl.specialize(key_cache.size(2))
    page_size = hl.specialize(key_cache.size(1))
    num_queries_per_kv = num_heads // num_kv_heads
    assert page_size == value_cache.size(1)
    assert head_dim == key_cache.size(3) == value_cache.size(3)

    out = torch.empty_like(q)

    # pyrefly: ignore[bad-argument-type]
    block_q = hl.register_block_size(1, q_block_padded_size)

    num_seqs = seq_lens.size(0)
    for seq_tile, head_tile in hl.tile([num_seqs, num_heads], block_size=[1, 1]):
        seq = seq_tile.begin
        head = head_tile.begin
        kv_head = head // num_queries_per_kv

        seq_len = seq_lens[seq]
        query_start = cu_seqlens_q[seq]
        query_end = cu_seqlens_q[seq + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        # exact loop over this sequence's query tokens; the tile's implicit
        # bounds mask drops the padding rows of the last tile, in the q load
        # and the out store alike
        for tok_tile in hl.tile(query_start, query_end, block_size=block_q):
            qt = q[tok_tile, head, :]
            q_pos = tok_tile.index - query_start

            m_i = hl.full([block_q], float("-inf"), dtype=torch.float32)
            l_i = hl.full([block_q], 1.0, dtype=torch.float32)
            acc = hl.zeros([block_q, head_dim], dtype=torch.float32)

            # causal bound: no row of this tile sees kv beyond context_len +
            # (the tile's last query position) + 1
            kv_bound = torch.minimum(
                context_len + tok_tile.begin + block_q - query_start, seq_len
            )
            num_blocks = (kv_bound + page_size - 1) // page_size

            # first kv position not visible to *every* row of the tile;
            # pages entirely below it need neither causal mask nor seq_len
            # zero-fill, so the page loop splits in two. No NaN hazard: if
            # full_pages > 0 every row enters the boundary loop with finite
            # m_i; otherwise the first boundary page contains kv position 0,
            # which is visible to every row.
            diag_start = context_len + tok_tile.begin - query_start + 1
            full_pages = torch.minimum(diag_start // page_size, num_blocks)

            # ── full pages: no causal mask, no seq_len zero-fill ──────────
            for page_idx in hl.grid(0, full_pages):
                page = block_table[seq, page_idx]
                k = key_cache[page, :, kv_head, :]
                v = value_cache[page, :, kv_head, :]

                scores = hl.dot(qt, k.transpose(0, 1), out_dtype=torch.float32)
                scores = scores * scale

                m_new = torch.maximum(m_i, torch.amax(scores, 1))
                p = torch.exp2(scores - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, 1)
                acc = acc * alpha[:, None]
                acc = hl.dot(p.to(v.dtype), v, out_dtype=torch.float32, acc=acc)
                m_i = m_new

            # ── boundary pages around the diagonal: masked body ──────────
            for page_idx in hl.grid(full_pages, num_blocks):
                page = block_table[seq, page_idx]
                k = key_cache[page, :, kv_head, :]
                v = value_cache[page, :, kv_head, :]
                # absolute kv position of every column; zero-fill past
                # seq_len so the partial last page is harmless
                kv_pos = page_idx * page_size + hl.arange(page_size)
                kv_valid = kv_pos < seq_len
                pad = hl.zeros([page_size, head_dim], dtype=k.dtype)
                k = torch.where(kv_valid[:, None], k, pad)
                v = torch.where(kv_valid[:, None], v, pad)

                scores = hl.dot(qt, k.transpose(0, 1), out_dtype=torch.float32)
                scores = scores * scale
                # causal mask: query position i attends to kv < context_len + i + 1
                scores = torch.where(
                    kv_pos[None, :] < context_len + q_pos[:, None] + 1,
                    scores,
                    float("-inf"),
                )

                m_new = torch.maximum(m_i, torch.amax(scores, 1))
                p = torch.exp2(scores - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, 1)
                acc = acc * alpha[:, None]
                acc = hl.dot(p.to(v.dtype), v, out_dtype=torch.float32, acc=acc)
                m_i = m_new

            out[tok_tile, head, :] = (acc / l_i[:, None]).to(q.dtype)
    return out


# %%
def paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """Computes causal attention over a paged KV cache."""
    return paged_attention_kernel(
        q,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        sm_scale * LOG2E,
        # ladder: one autotune bucket per power-of-two query length. It is
        # necessary because decode and prefill batches want opposite tile
        # shapes, so a single shared config would fit neither, and because
        # the bucket is also the ceiling of the query tile's width — without
        # raising it, long prefills cannot reach wide query tiles.
        min(MAX_Q_BLOCK_PADDED, _next_power_of_2(max_query_len)),
    )


# %%
# Reference Implementation
# ------------------------


# %%
def paged_attention_reference(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """Gathers each sequence's KV from its pages and runs SDPA per sequence."""
    page_size = key_cache.size(1)
    num_queries_per_kv = q.size(1) // key_cache.size(2)
    out = torch.empty_like(q)
    for s in range(seq_lens.numel()):
        start = int(cu_seqlens_q[s])
        end = int(cu_seqlens_q[s + 1])
        seq_len = int(seq_lens[s])
        num_pages = (seq_len + page_size - 1) // page_size
        pages = block_table[s, :num_pages].long()
        # [num_kv_heads, seq_len, head_dim], expanded to query heads for GQA
        k = key_cache[pages].flatten(0, 1)[:seq_len].transpose(0, 1).float()
        v = value_cache[pages].flatten(0, 1)[:seq_len].transpose(0, 1).float()
        k = k.repeat_interleave(num_queries_per_kv, dim=0)
        v = v.repeat_interleave(num_queries_per_kv, dim=0)
        qs = q[start:end].transpose(0, 1).float()  # [num_heads, q_len, head_dim]
        # query position i attends to kv positions <= context_len + i
        q_len = end - start
        q_pos = torch.arange(q_len, device=q.device)
        kv_pos = torch.arange(seq_len, device=q.device)
        mask = kv_pos[None, :] <= (seq_len - q_len) + q_pos[:, None]
        o = torch.nn.functional.scaled_dot_product_attention(
            qs, k, v, attn_mask=mask, scale=sm_scale
        )
        out[start:end] = o.transpose(0, 1).to(q.dtype)
    return out


# %%
# Testing
# -------


# %%
def test(
    query_lens: list[int] | None = None,
    seq_lens: list[int] | None = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> None:
    """Runs the kernel against the reference on the given batch."""
    # helion._testing hard-imports pytest; importing it lazily keeps this
    # module (and the kernel) importable without a test framework installed,
    # e.g. when a benchmark harness loads the kernel by file path.
    from helion._testing import DEVICE
    from helion._testing import HALF_DTYPE
    from helion._testing import run_example

    dtype = HALF_DTYPE if dtype is None else dtype
    device = DEVICE if device is None else device
    torch.manual_seed(0)
    # default batch: decodes with context, prefills with and without
    # context, and a ragged length
    if query_lens is None or seq_lens is None:
        query_lens = [1, 1, 1, 128, 512, 2048, 333, 1]
        seq_lens = [4096, 1023, 77, 512, 1024, 2048, 1000, 3]
    assert seq_lens is not None and len(seq_lens) == len(query_lens)

    cu_seqlens_q = [0]
    for n in query_lens:
        cu_seqlens_q.append(cu_seqlens_q[-1] + n)
    q = torch.randn(cu_seqlens_q[-1], num_heads, head_dim, dtype=dtype, device=device)

    # page pool with a disjoint page range per sequence
    pages_per_seq = [(n + page_size - 1) // page_size for n in seq_lens]
    key_cache = torch.randn(
        sum(pages_per_seq),
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(
        len(seq_lens), max(pages_per_seq), dtype=torch.int32, device=device
    )
    page_start = 0
    for row, n in enumerate(pages_per_seq):
        block_table[row, :n] = torch.arange(page_start, page_start + n, device=device)
        page_start += n

    args = (
        q,
        key_cache,
        value_cache,
        block_table,
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
        max(query_lens),
        head_dim**-0.5,
    )
    run_example(paged_attention, paged_attention_reference, args)


# %%
# Main Function
# ------------


# %%
def main() -> None:
    """Runs the example on a mixed batch and a pure-decode batch."""
    test()
    # pure decode: the ladder keys this batch at q_block_padded_size == 1,
    # so it autotunes separately from the prefill batch above
    test(query_lens=[1] * 32, seq_lens=[1024 + 7 * i for i in range(32)])


if __name__ == "__main__":
    main()
