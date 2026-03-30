#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from huggingface_hub import InferenceClient

token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    print("环境变量 HUGGINGFACE_TOKEN 未设置！")
    sys.exit(1)


def text_to_image(prompt: str, model: str = "black-forest-labs/FLUX.1-schnell", output_path: str = None):
    """
    文字生成图片功能
    
    参数:
        prompt: 图片描述文本（英文效果更好）
        model: 使用的模型，默认为 black-forest-labs/FLUX.1-schnell
        output_path: 输出图片路径，默认为当前目录下的时间戳命名
    
    常用文生图模型:
        black-forest-labs/FLUX.1-schnell - FLUX 快速版，推荐
        black-forest-labs/FLUX.1-dev - FLUX 开发版，质量更高但慢
        stabilityai/stable-diffusion-3.5-large - Stable Diffusion 3.5
        stabilityai/stable-diffusion-xl-base-1.0 - SDXL
    
    返回:
        保存的图片路径
    """
    
    client = InferenceClient(
        provider="nebius",
        api_key=token,
        timeout=120,  # 图片生成可能需要更长时间
    )
    
    print(f"调用模型: {model}")
    print(f"提示词: {prompt}")
    print("正在生成图片，请稍候...")
    
    # output is a PIL.Image object
    image = client.text_to_image(
        prompt,
        model=model,
    )
    
    # 如果没有指定输出路径，使用时间戳命名
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_image_{timestamp}.png"
    
    # 保存图片
    image.save(output_path)
    print(f" 图片已保存到: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # 测试代码
    test_prompts = [
        "Astronaut riding a horse",
        "A beautiful sunset over the ocean",
        "A cute cat playing with a ball of yarn",
    ]
    
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        print("使用测试提示词...")
        user_prompt = test_prompts[0]
    
    if user_prompt:
        try:
            saved_path = text_to_image(user_prompt)
            print(f"\n成功！图片保存在: {saved_path}")
        except Exception as e:
            print(f"\n生成图片失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有输入提示词！")
        print("用法: python src/TexttoImage.py '图片描述'")
