#!/usr/bin/env python3

import os
import sys
from huggingface_hub import InferenceClient

token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    print("环境变量 HUGGINGFACE_TOKEN 未设置！")
    sys.exit(1)


def textGeneration(prompt: str, model: str):
    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
        timeout=30,
    )
    

    print(f"调用模型: {model}")

    result = client.chat.completions.create(
        model=model,
            messages=[
        {
            "role": "user",
            "content": prompt
        }
        ],
    )
    output = result["choices"][0]["message"]["content"].strip()
    # print(f"结果:\n{output}")

    return output

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("请输入文本: ")
    
    if user_input:
        output = textGeneration(user_input, "HuggingFaceTB/SmolLM3-3B")
        print("\n模型返回：\n")
        print(output)
    else:
        print("没有输入")
    