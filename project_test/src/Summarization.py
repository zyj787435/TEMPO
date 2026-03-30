#!/usr/bin/env python3

import os
import sys
from huggingface_hub import InferenceClient

token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    print("环境变量 HUGGINGFACE_TOKEN 未设置！")
    sys.exit(1)


def summarization(text: str, model: str = "Falconsai/medical_summarization"):
    """
    文本摘要功能 
    参数:
        text: 需要总结的文本
        model: 使用的模型，默认为 Falconsai/medical_summarization
    
    返回:
        摘要文本
    """
    
    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
        timeout=30,
    )
    
    print(f"调用模型: {model}")
    
    result = client.summarization(
        text,
        model=model,
    )
    
    # result 是一个包含摘要的对象
    summary = result.summary_text if hasattr(result, 'summary_text') else str(result)
    
    return summary


if __name__ == "__main__":
    # 测试代码
    test_text = """The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""
    
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("使用测试文本...")
        user_input = test_text
    
    if user_input:
        summary = summarization(user_input)
        print("\n原文本摘要：\n")
        print(summary)
    else:
        print("没有输入")
