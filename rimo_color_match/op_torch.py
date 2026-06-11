import torch


def 图像相似度_torch(img_a: torch.Tensor, img_b: torch.Tensor, eps=1e-2) -> torch.Tensor:
    diff = torch.abs(img_a - img_b)
    match = (diff <= eps).all(dim=-1)
    return match.float().flatten(1).mean(dim=1)


def mma_clamp_torch(A, B, C):
    t = torch.matmul(A, B) + C.unsqueeze(1)
    return torch.clamp(t, 0.0, 1.0)
