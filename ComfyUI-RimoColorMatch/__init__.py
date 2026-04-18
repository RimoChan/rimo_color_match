import torch


class RimoColorMatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "search_steps": ("INT", {"default": 2000, "min": 1, "max": 100000, "step": 1, "display": "number"}),
                "d": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001, "display": "number"}),
                "size": ("INT", {"default": 128, "min": 16, "max": 1024, "step": 8, "display": "number"}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "batch_size": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "display": "number"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_image",)
    
    FUNCTION = "process"
    
    CATEGORY = "image/color_processing"

    def process(self, image_ref, image_target, search_steps, d, size, seed, batch_size):
        from rimo_color_match import 匹配颜色
        result_images = []
        for i in range(image_ref.shape[0]):
            matched_single = 匹配颜色(
                img_a=image_target[i].cuda(),
                img_b=image_ref[i].cuda(),
                搜索次数=search_steps,
                d=d,
                size=size,
                seed=seed,
                batch_size=batch_size
            )[0]
            result_images.append(matched_single)
        out_image = torch.stack(result_images, dim=0)
        return out_image,


NODE_CLASS_MAPPINGS = {
    "RimoColorMatch": RimoColorMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RimoColorMatch": "色色匹配",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
