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

Structure of the kernel (GQA supported; the rows of a query tile never
straddle KV heads):

- **Flat q-block grid**: programs are indexed by a global q-block id and
  resolve their sequence through a vectorized rank reduction over
  ``cu_seqlens_q`` instead of a dense ``[num_seqs, heads, max_query_len]``
  grid, which over-provisions for skewed batches. Because the sequence index
  is computed on the device, the block table is passed flattened with an
  explicit row stride.
- **Reshape-free tiles**: the wrapper flattens ``q`` and the caches into 2-D
  views (free for contiguous tensors), and every tile is loaded in its final
  shape. A query tile is ``[block_m, head_dim]``, where row ``m`` is the
  (token, head) pair ``query_start + m // group`` / ``tile_h.begin + m %
  group``; a KV tile is ``[block_n, head_dim]`` gathered through computed
  flat column ids. No reshape, flatten or repeat ever merges tile
  dimensions: such a merge moves elements between threads, which the CuTe
  backend's SIMT lowering refuses for any consumer that reads the result at
  its new position, while 2-D loads, gathers, transposes and dots lower on
  every backend.
- **Active contraction tile**: the scores dot contracts ``head_dim``
  through its own tile loop (``block_k``, equal to ``head_dim`` in the
  default configs), instead of a single dot over a free static extent: the
  CuTe dot lowering requires an active tile on the contraction axis, and
  both operands are re-read per chunk, which is also the CuTe execution
  model. Triton lowers both forms identically up to fp reassociation.
- **Online (flash-style) softmax** with ``exp2`` (``log2(e)`` is folded into
  the scale host-side) over KV tiles gathered page by page.
- **KV loop split** at the causal diagonal: column tiles entirely below the
  diagonal are visible to every query row of the block, so they run without
  the causal mask and the ``seq_len`` zero-fill; only the boundary tiles
  around the diagonal carry masks. Each loop gets its own block size.
- **Ladder specialization**: ``q_rows_padded_size`` is the query length
  rounded up to a power of two (times the GQA group), so decode and prefill
  batches of different lengths autotune separately. At the same time, the
  ladder is required to not compile & autotune every different shape that
  happens to be scheduled by vLLM.
- **Default configs**: a small list of tile shapes covering the serving
  regimes. The autotuner measures them per specialization key and keeps
  the fastest, which keeps the example quick and deterministic — but the
  shapes are informed by measurements of the source kernel on sm_100 GPUs
  and are not optimal on other targets or workloads. Remove ``configs=``
  from the decorator to let the autotuner search the full space.

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

# ceiling of the ladder, in query tokens (the kernel's ceiling is this
# times the GQA group, in packed query rows)
MAX_Q_BLOCK_PADDED = 8192


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


# Default tile shapes, measured on the source kernel: block_sizes are
# (query rows, boundary kv columns, full kv columns, head-dim chunk of the
# scores dot), where query rows are num_queries_per_kv * queries per tile,
# kv columns are pages * page_size, and the head-dim chunk must divide
# head_dim (block_k == head_dim restores the single-dot form). The
# autotuner measures these per specialization key and keeps the fastest;
# configs whose query tile exceeds the ladder ceiling of a key (the
# prefill shapes under a decode key) are dropped automatically.
#
# These shapes were tuned for the serving regimes above (GQA with
# num_queries_per_kv == 4, head_dim 128, page_size 16) and are not
# necessarily equally good for every batch: other head counts, head dims or
# page sizes shift both the ideal query tile and the KV tile, so a
# deployment should drop this list (letting the autotuner search the full
# space) or replace it with shapes measured on its own distribution.
DEFAULT_CONFIGS = [
    # prefill / mixed: wide query tile
    helion.Config(
        block_sizes=[128, 64, 64, 128], num_warps=4, num_stages=2, pid_type="flat"
    ),
    # short prefill: narrower KV tile
    helion.Config(
        block_sizes=[128, 32, 32, 128], num_warps=4, num_stages=2, pid_type="flat"
    ),
    # decode: minimal query tile, moderate KV tile
    helion.Config(
        block_sizes=[4, 64, 64, 128], num_warps=4, num_stages=2, pid_type="flat"
    ),
    # decode with a longer KV tile
    helion.Config(
        block_sizes=[4, 128, 128, 128], num_warps=4, num_stages=2, pid_type="flat"
    ),
]

# %%
# Paged Attention Kernel
# ----------------------


# %%
# Based on the kernel presented in the PyTorch blog post "Portable paged attention
# in Helion" (https://pytorch.org/blog/portable-paged-attention-in-helion/).
# Unlike a naive implementation, which materializes each sequence's full
# [q, kv] score matrix, this kernel runs an online softmax over one KV tile
# at a time and drops masks on tiles entirely below the causal diagonal.
@helion.kernel(
    configs=DEFAULT_CONFIGS,
    static_shapes=False,
    index_dtype=torch.int64,
)
def paged_attention_kernel(
    q: torch.Tensor,  # [total_q_tokens * num_heads, head_dim]
    key_cache: torch.Tensor,  # [num_pages, page_size * num_kv_heads * head_dim]
    value_cache: torch.Tensor,  # same layout as key_cache
    block_tables_flat: torch.Tensor,  # [num_seqs * block_table_stride]
    block_table_stride: int,
    seq_lens: torch.Tensor,  # [num_seqs]
    cu_seqlens_q: torch.Tensor,  # [num_seqs + 1] query-token offsets
    scale: float,  # sm_scale * LOG2E
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    page_size: int,
    q_rows_padded_size: hl.constexpr,
    num_seqs_padded: hl.constexpr,
) -> torch.Tensor:
    """
    Computes causal attention over a paged KV cache; see the module docstring.

    Args:
        q: Packed queries flattened to rows ``token * num_heads + head``.
        key_cache: Pooled keys, each page flattened to one row of
            ``page_size * num_kv_heads * head_dim`` elements.
        value_cache: Pooled values, same layout as ``key_cache``.
        block_tables_flat: The block table flattened to one row per sequence
            of length ``block_table_stride``.
        seq_lens: Total (context + query) length of each sequence.
        cu_seqlens_q: Exclusive scan of the per-sequence query lengths.
        scale: Softmax scale pre-multiplied by ``log2(e)``.
        num_seqs: Number of sequences in the batch.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads; divides ``num_heads``.
        page_size: Token rows of the KV cache per page.
        q_rows_padded_size: Ladder bucket, ``next_power_of_2(max_query_len)
            * num_queries_per_kv``; the ceiling of the query tile's rows.
        num_seqs_padded: Width of the ``gid -> seq_idx`` reduction vector.

    Returns:
        Output tensor of shape [total_q_tokens * num_heads, head_dim].
    """
    num_heads = hl.specialize(num_heads)
    num_kv_heads = hl.specialize(num_kv_heads)
    page_size = hl.specialize(page_size)
    head_dim = hl.specialize(q.size(1))
    kv_row_stride = hl.specialize(key_cache.size(1))
    num_queries_per_kv = num_heads // num_kv_heads
    assert page_size * num_kv_heads * head_dim == kv_row_stride == value_cache.size(1)
    # block sizes are powers of two, so every head-dim chunk of the scores
    # dot must divide head_dim exactly: a partial chunk would feed masked
    # load fills into the dot
    assert (head_dim & (head_dim - 1)) == 0

    out = torch.empty_like(q)

    # pyrefly: ignore[bad-argument-type]
    block_m = hl.register_block_size(1, q_rows_padded_size)
    block_n = hl.register_block_size(1, 32 * page_size)
    block_n_full = hl.register_block_size(1, 32 * page_size)
    block_k = hl.register_block_size(min(16, head_dim), head_dim)

    # upper bound on the total number of q blocks: each sequence contributes
    # at most its query rows // block_m + 1 blocks
    num_q_blocks = q.size(0) // num_kv_heads // block_m + num_seqs

    for q_blk_tile, tile_h in hl.tile(
        [num_q_blocks, num_heads], block_size=[1, num_queries_per_kv]
    ):
        gid = q_blk_tile.begin

        # vectorized gid -> seq_idx: cu_seqlens_q * group // block_m + s is
        # monotone in s, so the sequence owning this block is the rank of
        # gid among the per-sequence block starts
        s_idx = hl.arange(num_seqs_padded)  # pyrefly: ignore[bad-argument-type]
        valid = s_idx < num_seqs
        cu = hl.load(cu_seqlens_q, [s_idx], extra_mask=valid)
        starts = (cu * num_queries_per_kv) // block_m + s_idx
        seq_idx = torch.sum(((starts <= gid) & valid).to(torch.int32)) - 1

        seq_len = seq_lens[seq_idx]
        query_start = cu_seqlens_q[seq_idx]
        query_end = cu_seqlens_q[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len
        # row offset (in packed query rows) of this q block within its
        # sequence
        q_row_offset = (
            gid - ((query_start * num_queries_per_kv) // block_m + seq_idx)
        ) * block_m
        # flat row offset into the block table; the 2-D form cannot be
        # lowered when seq_idx is a device-computed scalar
        block_table_offset = seq_idx * block_table_stride

        # blocks past this sequence's last row are padding
        if q_row_offset < query_len * num_queries_per_kv:
            kv_head = tile_h.begin // num_queries_per_kv

            # query rows for the scores dot: row m is the (token, head)
            # pair of the group, i.e. token query_start + m // group and
            # head tile_h.begin + m % group; q is loaded per head-dim chunk
            # inside the KV loops
            row = q_row_offset + hl.arange(block_m)
            q_pos = row // num_queries_per_kv
            token = query_start + q_pos
            q_flat_row = (
                token * num_heads + tile_h.begin + row % num_queries_per_kv
            )
            q_mask = token < query_end

            m_i = hl.full([block_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([block_m], 1.0, dtype=torch.float32)
            acc = hl.zeros([block_m, head_dim], dtype=torch.float32)

            # causal bound: no row of this q block sees kv beyond
            # context_len + (the block's last query position) + 1
            kv_bound = torch.minimum(
                context_len
                + (q_row_offset + block_m - 1) // num_queries_per_kv
                + 1,
                seq_len,
            )
            num_blocks = (kv_bound + page_size - 1) // page_size
            kv_end = num_blocks * page_size

            # first kv position not visible to *every* row of the q block;
            # column tiles entirely below it need neither causal mask nor
            # seq_len zero-fill, so the KV loop splits in two. No NaN hazard:
            # if full_cols > 0 every row enters the boundary loop with
            # finite m_i; otherwise the first boundary tile contains kv
            # position 0, which is visible to every row.
            diag_start = context_len + q_row_offset // num_queries_per_kv + 1
            full_cols = (diag_start // block_n_full) * block_n_full
            full_cols = torch.minimum(full_cols, kv_end)

            # Splitting the loop is a hand-applied optimization for GPU-like
            # targets. It pays off here because the masked body costs two
            # extra ``torch.where`` passes over the k/v tiles plus the mask
            # On a target where predication is close to free, or where a
            # second loop nest is expensive (extra prologue, instruction
            # cache, fixed tile shapes of a matmul unit), the single masked
            # loop can be superiror. Future compiler passes could automate
            # loop splits like this (since the range is monotonic).
            # ── full tiles: no causal mask, no seq_len zero-fill ──────────
            for tile_n in hl.tile(full_cols, block_size=block_n_full):
                kv_pos = tile_n.begin + hl.arange(block_n_full)
                # every column is below the diagonal and thus inside the
                # sequence's pages, so the page lookup is always in-row
                page = hl.load(
                    block_tables_flat,
                    [block_table_offset + kv_pos // page_size],
                )
                page = page.view([block_n_full]).to(torch.int64)
                col = (
                    page * kv_row_stride
                    + (kv_pos % page_size) * (num_kv_heads * head_dim)
                    + kv_head * head_dim
                )
                v = hl.load(value_cache, [col, hl.arange(head_dim)])

                # the scores dot contracts head_dim in chunks of an active
                # tile loop: the CuTe dot lowering requires the contraction
                # dim to be an active tile (block_k == head_dim restores the
                # single-dot form), and both operands are re-read per chunk
                scores = hl.zeros([block_m, block_n_full], dtype=torch.float32)
                for tile_d in hl.tile(head_dim, block_size=block_k):
                    qk = hl.load(
                        q, [q_flat_row, tile_d], extra_mask=q_mask[:, None]
                    )
                    kk = hl.load(key_cache, [col, tile_d])
                    scores = hl.dot(
                        qk,
                        kk.transpose(0, 1),
                        out_dtype=torch.float32,
                        acc=scores,
                    )
                scores = scores * scale

                m_new = torch.maximum(m_i, torch.amax(scores, 1))
                p = torch.exp2(scores - m_new[:, None])
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + torch.sum(p, 1)
                acc = acc * alpha[:, None]
                acc = hl.dot(p.to(v.dtype), v, out_dtype=torch.float32, acc=acc)
                m_i = m_new

            # ── boundary tiles around the diagonal: masked body ──────────
            for tile_n in hl.tile(full_cols, kv_end, block_size=block_n):
                kv_pos = tile_n.begin + hl.arange(block_n)
                # the last tile can overshoot the sequence's pages; clamp
                # the page lookup into the sequence's valid pages (the
                # overshoot columns are zero-filled below)
                page = hl.load(
                    block_tables_flat,
                    [
                        block_table_offset
                        + torch.minimum(
                            kv_pos // page_size, (seq_len - 1) // page_size
                        )
                    ],
                )
                page = page.view([block_n]).to(torch.int64)
                col = (
                    page * kv_row_stride
                    + (kv_pos % page_size) * (num_kv_heads * head_dim)
                    + kv_head * head_dim
                )
                v = hl.load(value_cache, [col, hl.arange(head_dim)])
                # absolute kv position of every column; zero-fill past
                # seq_len so the partial last page and the clamped tail of
                # the block table are harmless
                kv_valid = kv_pos < seq_len
                pad = hl.zeros([block_n, head_dim], dtype=v.dtype)
                v = torch.where(kv_valid[:, None], v, pad)

                scores = hl.zeros([block_m, block_n], dtype=torch.float32)
                for tile_d in hl.tile(head_dim, block_size=block_k):
                    qk = hl.load(
                        q, [q_flat_row, tile_d], extra_mask=q_mask[:, None]
                    )
                    kk = hl.load(key_cache, [col, tile_d])
                    pad_k = hl.zeros([block_n, block_k], dtype=kk.dtype)
                    kk = torch.where(kv_valid[:, None], kk, pad_k)
                    scores = hl.dot(
                        qk,
                        kk.transpose(0, 1),
                        out_dtype=torch.float32,
                        acc=scores,
                    )
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

            acc = acc / l_i[:, None]
            hl.store(
                out,
                [q_flat_row, hl.arange(head_dim)],
                acc,
                extra_mask=q_mask[:, None],
            )
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
    """
    Computes causal attention over a paged KV cache.

    Same argument contract as :func:`paged_attention_reference`; flattens
    ``q`` and the caches into the 2-D layouts the kernel gathers from (free
    views of contiguous tensors) and views the result back to ``q``'s shape.
    """
    num_queries_per_kv = q.size(1) // key_cache.size(2)
    q_rows_padded = min(MAX_Q_BLOCK_PADDED, _next_power_of_2(max_query_len))
    out = paged_attention_kernel(
        q.reshape(-1, q.size(2)),
        key_cache.reshape(key_cache.size(0), -1),
        value_cache.reshape(value_cache.size(0), -1),
        # seq_idx is device-computed, so the block table travels flattened
        block_table.reshape(-1),
        block_table.stride(0),
        seq_lens,
        cu_seqlens_q,
        sm_scale * LOG2E,
        seq_lens.size(0),
        q.size(1),
        key_cache.size(2),
        key_cache.size(1),
        # ladder: one autotune bucket per power-of-two query length. It is
        # necessary because decode and prefill batches want opposite tile
        # shapes, so a single shared config would fit neither, and because
        # the bucket is also the ceiling of the query tile's rows — without
        # raising it, long prefills cannot reach wide query tiles.
        q_rows_padded_size=q_rows_padded * num_queries_per_kv,
        num_seqs_padded=_next_power_of_2(seq_lens.size(0)),
    )
    return out.view(q.shape)


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
    # pure decode: the ladder keys this batch at q_rows_padded_size == 1 *
    # num_queries_per_kv, so it autotunes separately from the prefill batch
    # above
    test(query_lens=[1] * 32, seq_lens=[1024 + 7 * i for i in range(32)])


if __name__ == "__main__":
    main()
