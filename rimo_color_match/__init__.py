from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image


def 图像相似度(img_a: torch.Tensor, img_b: torch.Tensor, eps=1e-2) -> float:
    diff = torch.abs(img_a - img_b)
    match = (diff <= eps).all(dim=-1)
    return match.float().flatten(1).mean(dim=1)


def 匹配颜色(img_a: torch.Tensor, img_b: torch.Tensor, 搜索次数=2000, d=0.01, size=128, seed=1, batch_size=512) -> torch.Tensor:
    device = img_a.device

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    高, 宽, C = img_a.shape

    img_a_original_flat = img_a.view(-1, 3)

    if max(高, 宽) > size:
        scale = size / max(高, 宽)
        new高 = int(高 * scale)
        new宽 = int(宽 * scale)
        img_a = F.interpolate(img_a.permute(2, 0, 1).unsqueeze(0), size=(new高, new宽), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        img_b = F.interpolate(img_b.permute(2, 0, 1).unsqueeze(0), size=(new高, new宽), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

    img_a_flat = img_a.view(-1, 3)
    img_b_flat = img_b.view(-1, 3)

    当前_W = torch.eye(3, device=device)
    当前_B = torch.zeros(3, device=device)
    
    最大图像相似度 = 图像相似度(img_a.unsqueeze(0), img_b)
    for _ in tqdm(range(搜索次数), desc="搜索颜色变换矩阵"):
        dW = d * torch.randn(batch_size, 3, 3, device=device, generator=generator)
        dB = d * torch.randn(batch_size, 3, device=device, generator=generator)
        
        测试_W = 当前_W + dW
        测试_B = 当前_B + dB
        
        img_b2_flat = torch.matmul(img_a_flat, 测试_W) + 测试_B.unsqueeze(1)
        img_b2_flat = torch.clamp(img_b2_flat, 0.0, 1.0)
        sims = 图像相似度(img_b_flat.unsqueeze(0), img_b2_flat)
        max_sim, max_idx = torch.max(sims, dim=0)
        t = max_sim.item()
        if t > 最大图像相似度:
            最大图像相似度 = t
            print('更新', 最大图像相似度)
            当前_W = 测试_W[max_idx]
            当前_B = 测试_B[max_idx]

    print(f"{最大图像相似度=}")
    print("最佳 W 矩阵:\n", 当前_W.cpu().numpy())
    print("最佳 B 向量:\n", 当前_B.cpu().numpy())

    最终_img = torch.clamp(torch.matmul(img_a_original_flat, 当前_W) + 当前_B, 0.0, 1.0).view(高, 宽, 3)
    return 最终_img, 当前_W, 当前_B


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    a = read_image('Flux.png').permute(1, 2, 0).to(torch.float32).to(device) / 255.0
    b = read_image('原图.png').permute(1, 2, 0).to(torch.float32).to(device) / 255.0

    c, W, B = 匹配颜色(a, b)
    save_image(c.permute(2, 0, 1), 'output.png')
