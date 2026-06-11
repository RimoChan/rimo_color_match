import time
import torch

import time
import torch
import triton
import triton.language as tl


@triton.jit
def image_similarity_kernel(
    a_ptr, b_ptr, out_ptr,
    N, C, eps,
    stride_a_b, stride_a_n, stride_a_c,
    stride_b_b, stride_b_n, stride_b_c,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offsets = tl.arange(0, BLOCK_C)

    n_mask = n_offsets < N
    c_mask = c_offsets < C
    mask = n_mask[:, None] & c_mask[None, :]

    a_ptrs = a_ptr + pid_b * stride_a_b + n_offsets[:, None] * stride_a_n + c_offsets[None, :] * stride_a_c
    b_ptrs = b_ptr + pid_b * stride_b_b + n_offsets[:, None] * stride_b_n + c_offsets[None, :] * stride_b_c

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    diff = tl.abs(a - b)
    is_match = tl.min(diff <= eps, axis=1)

    valid_pixel_match = is_match & n_mask
    block_matches = tl.sum(valid_pixel_match, axis=0, dtype=tl.uint16)

    if block_matches > 0:
        tl.atomic_add(out_ptr + pid_b, block_matches)


def 图像相似度_triton(img_a: torch.Tensor, img_b: torch.Tensor, eps=1e-2) -> torch.Tensor:
    B1, N, C = img_a.shape
    B2, _, _ = img_b.shape
    assert B1 == B2 or B1 == 1 or B2 == 1
    B = max(B1, B2)

    img_a = img_a.contiguous()
    img_b = img_b.contiguous()

    a_view = img_a.view(-1, N, C)
    b_view = img_b.view(-1, N, C)

    out = torch.zeros(B, dtype=torch.int32, device=img_a.device)

    BLOCK_C = triton.next_power_of_2(C)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), B)

    stride_a_b = a_view.stride(0) if B1 > 1 else 0
    stride_b_b = b_view.stride(0) if B2 > 1 else 0

    image_similarity_kernel[grid](
        a_view, b_view, out,
        N, C, eps,
        stride_a_b, a_view.stride(1), a_view.stride(2),
        stride_b_b, b_view.stride(1), b_view.stride(2),
        BLOCK_N=4096,
        BLOCK_C=BLOCK_C
    )
    return out / N


# 下面这个函数是Gemini写的，我也没想到要这样写，但是它是真的快……
@triton.jit
def _mma_clamp_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    N,
    stride_out_b, stride_out_n,
    stride_b_b, stride_c_b,
    BLOCK_N: tl.constexpr
):
    # 获取程序块的索引
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    # 计算 N 维度上的数据偏移和掩码
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # 1. 连续加载 A 矩阵 (N x 3)
    # 因为所有 Batch 共享同一份 A，所以这里会被硬件 L2 Cache 完美缓存
    a_base = a_ptr + offs_n * 3
    a0 = tl.load(a_base + 0, mask=mask_n)
    a1 = tl.load(a_base + 1, mask=mask_n)
    a2 = tl.load(a_base + 2, mask=mask_n)

    # 2. 加载当前 Batch 的 B 矩阵 (3 x 3) 作为标量
    b_base = b_ptr + pid_b * stride_b_b
    b00 = tl.load(b_base + 0)
    b01 = tl.load(b_base + 1)
    b02 = tl.load(b_base + 2)
    b10 = tl.load(b_base + 3)
    b11 = tl.load(b_base + 4)
    b12 = tl.load(b_base + 5)
    b20 = tl.load(b_base + 6)
    b21 = tl.load(b_base + 7)
    b22 = tl.load(b_base + 8)

    # 3. 加载当前 Batch 的 C 向量 (3) 作为标量
    c_base = c_ptr + pid_b * stride_c_b
    c0 = tl.load(c_base + 0)
    c1 = tl.load(c_base + 1)
    c2 = tl.load(c_base + 2)

    # 4. 手动展开矩阵乘法 (A @ B) + C
    out0 = a0 * b00 + a1 * b10 + a2 * b20 + c0
    out1 = a0 * b01 + a1 * b11 + a2 * b21 + c1
    out2 = a0 * b02 + a1 * b12 + a2 * b22 + c2

    # 5. 融合 Clamp 操作
    out0 = tl.maximum(0.0, tl.minimum(1.0, out0))
    out1 = tl.maximum(0.0, tl.minimum(1.0, out1))
    out2 = tl.maximum(0.0, tl.minimum(1.0, out2))

    # 6. 写回显存 (B, N, 3)
    out_base = out_ptr + pid_b * stride_out_b + offs_n * stride_out_n
    tl.store(out_base + 0, out0, mask=mask_n)
    tl.store(out_base + 1, out1, mask=mask_n)
    tl.store(out_base + 2, out2, mask=mask_n)


# A.shape: torch.Size([N, 3])
# B.shape: torch.Size([B, 3, 3])
# C.shape: torch.Size([B, 3])
def mma_clamp_triton(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    N = A.shape[0]
    batch_size = B.shape[0]

    out = torch.empty((batch_size, N, 3), device=A.device, dtype=A.dtype)

    BLOCK_N = 512
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), batch_size)

    _mma_clamp_kernel[grid](
        A, B, C, out,
        N,
        out.stride(0), out.stride(1),
        B.stride(0), C.stride(0),
        BLOCK_N=BLOCK_N,
    )
    return out


if __name__ == '__main__':
    from op_torch import 图像相似度_torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    b = torch.randn([512, 128*80, 3], device='cuda:0') / 200
    a = torch.randn([1, 128*80, 3], device='cuda:0') / 200

    for _ in range(3):
        s = []
        for 函数 in [图像相似度_triton, 图像相似度_torch]:
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(10):
                out = 函数(a, b)
            s.append(out)
            torch.cuda.synchronize()
            print(函数.__name__, f'耗时', time.time() - st)
        print(s[0].equal(s[1]))
