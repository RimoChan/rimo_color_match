import time
import torch
import triton
import triton.language as tl


@triton.jit
def _mma_clamp_sim_kernel(
    a_ptr, b_ptr, w_ptr, bias_ptr, out_ptr,
    batch_size, N, eps, inv_N,
    stride_w_b, stride_bias_b,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offs < batch_size

    w_base = w_ptr + b_offs * stride_w_b
    w00 = tl.load(w_base + 0, mask=b_mask, other=0.0)[:, None]
    w01 = tl.load(w_base + 1, mask=b_mask, other=0.0)[:, None]
    w02 = tl.load(w_base + 2, mask=b_mask, other=0.0)[:, None]
    w10 = tl.load(w_base + 3, mask=b_mask, other=0.0)[:, None]
    w11 = tl.load(w_base + 4, mask=b_mask, other=0.0)[:, None]
    w12 = tl.load(w_base + 5, mask=b_mask, other=0.0)[:, None]
    w20 = tl.load(w_base + 6, mask=b_mask, other=0.0)[:, None]
    w21 = tl.load(w_base + 7, mask=b_mask, other=0.0)[:, None]
    w22 = tl.load(w_base + 8, mask=b_mask, other=0.0)[:, None]

    c_base = bias_ptr + b_offs * stride_bias_b
    c0 = tl.load(c_base + 0, mask=b_mask, other=0.0)[:, None]
    c1 = tl.load(c_base + 1, mask=b_mask, other=0.0)[:, None]
    c2 = tl.load(c_base + 2, mask=b_mask, other=0.0)[:, None]

    acc = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.int32)
    offs_base = tl.arange(0, BLOCK_N)
    full_end = (N // BLOCK_N) * BLOCK_N

    for n in range(0, full_end, BLOCK_N):
        idx = (n + offs_base) * 3
        a0 = tl.load(a_ptr + idx)[None, :]
        a1 = tl.load(a_ptr + idx + 1)[None, :]
        a2 = tl.load(a_ptr + idx + 2)[None, :]
        t0 = tl.load(b_ptr + idx)[None, :]
        t1 = tl.load(b_ptr + idx + 1)[None, :]
        t2 = tl.load(b_ptr + idx + 2)[None, :]

        o0 = tl.clamp(a0 * w00 + a1 * w10 + a2 * w20 + c0, 0.0, 1.0)
        o1 = tl.clamp(a0 * w01 + a1 * w11 + a2 * w21 + c1, 0.0, 1.0)
        o2 = tl.clamp(a0 * w02 + a1 * w12 + a2 * w22 + c2, 0.0, 1.0)

        m = (tl.abs(o0 - t0) <= eps) & (tl.abs(o1 - t1) <= eps) & (tl.abs(o2 - t2) <= eps)
        acc += m.to(tl.int32)

    if full_end < N:
        offs = full_end + offs_base
        n_mask = offs < N
        idx = offs * 3
        a0 = tl.load(a_ptr + idx, mask=n_mask, other=0.0)[None, :]
        a1 = tl.load(a_ptr + idx + 1, mask=n_mask, other=0.0)[None, :]
        a2 = tl.load(a_ptr + idx + 2, mask=n_mask, other=0.0)[None, :]
        t0 = tl.load(b_ptr + idx, mask=n_mask, other=0.0)[None, :]
        t1 = tl.load(b_ptr + idx + 1, mask=n_mask, other=0.0)[None, :]
        t2 = tl.load(b_ptr + idx + 2, mask=n_mask, other=0.0)[None, :]

        o0 = tl.clamp(a0 * w00 + a1 * w10 + a2 * w20 + c0, 0.0, 1.0)
        o1 = tl.clamp(a0 * w01 + a1 * w11 + a2 * w21 + c1, 0.0, 1.0)
        o2 = tl.clamp(a0 * w02 + a1 * w12 + a2 * w22 + c2, 0.0, 1.0)

        m = (tl.abs(o0 - t0) <= eps) & (tl.abs(o1 - t1) <= eps) & (tl.abs(o2 - t2) <= eps) & n_mask[None, :]
        acc += m.to(tl.int32)

    total = tl.sum(acc, axis=1).to(tl.float32)
    tl.store(out_ptr + b_offs, total * inv_N, mask=b_mask)


def mma_clamp_图像相似度_triton(img_a: torch.Tensor, img_b: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps=1e-2) -> torch.Tensor:
    img_a = img_a.contiguous()
    img_b = img_b.contiguous()
    W = W.contiguous()
    B = B.contiguous()

    N, C = img_a.shape
    assert C == 3
    batch_size = W.shape[0]
    out = torch.empty(batch_size, dtype=torch.float32, device=img_a.device)

    _mma_clamp_sim_kernel[(triton.cdiv(batch_size, 2),)](
        img_a, img_b, W, B, out,
        batch_size, N, eps, 1.0 / N,
        W.stride(0), B.stride(0),
        BLOCK_B=2,
        BLOCK_N=256,
        num_warps=2,
    )
    return out


if __name__ == '__main__':
    from op_torch import mma_clamp_图像相似度_torch

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    a = torch.rand([128 * 80, 3], device='cuda:0')
    b = (a * 0.9 + 0.05).clamp(0.0, 1.0)
    W = torch.eye(3, device='cuda:0').unsqueeze(0) + 0.02 * torch.randn([512, 3, 3], device='cuda:0')
    B = 0.02 * torch.randn([512, 3], device='cuda:0')

    for _ in range(3):
        s = []
        for 函数 in [mma_clamp_图像相似度_triton, mma_clamp_图像相似度_torch]:
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(10):
                out = 函数(a, b, W, B)
            s.append(out)
            torch.cuda.synchronize()
            print(函数.__name__, '耗时', time.time() - st)
        print((s[0] - s[1]).abs().max().item())
