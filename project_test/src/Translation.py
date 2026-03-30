#!/usr/bin/env python3

import os
import sys
from huggingface_hub import InferenceClient

token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    print("环境变量 HUGGINGFACE_TOKEN 未设置！")
    sys.exit(1)


def translation(text: str, src_lang: str = "en_XX", tgt_lang: str = "zh_CN", model: str = "facebook/mbart-large-50-many-to-many-mmt"):
    """
    文本翻译功能
    
    参数:
        text: 需要翻译的文本
        model: 使用的模型，默认为 facebook/mbart-large-50-many-to-many-mmt
    
    支持的语言代码（mBART-50）:
        中文: zh_CN
        英语: en_XX
        日语: ja_XX
        韩语: ko_KR
        俄语: ru_RU
        法语: fr_XX
        德语: de_DE
        西班牙语: es_XX
        更多语言请查看: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
    
    返回:
        翻译后的文本
    """ 
    
    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
        timeout=30,
    )
    
    
    print(f"调用模型: {model}")
    
    result = client.translation(
        text,
        model=model,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    
    # result 是一个包含翻译文本的对象
    translated_text = result.translation_text if hasattr(result, 'translation_text') else str(result)
    
    return translated_text


if __name__ == "__main__":
    # 测试代码
    test_cases = [
        ("Hello, how are you?", "en_XX", "zh_CN"),  # 英语 -> 中文
        ("你好，世界！", "zh_CN", "en_XX"),  # 中文 -> 英语
        ("Меня зовут Вольфганг и я живу в Берлине", "ru_RU", "en_XX"),  # 俄语 -> 英语
    ]
    
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        # 默认英译中
        src_lang = "en_XX"
        tgt_lang = "zh_CN"
    else:
        print("使用测试文本（英语 -> 中文）...")
        user_input, src_lang, tgt_lang = test_cases[0]
    
    if user_input:
        translated = translation(user_input, src_lang, tgt_lang)
        print(f"\n原文: {user_input}")
        print(f"翻译: {translated}")
    else:
        print("没有输入")
