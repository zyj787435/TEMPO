#!/usr/bin/env python3

import os
import sys
from huggingface_hub import InferenceClient

token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    print("环境变量 HUGGINGFACE_TOKEN 未设置！")
    sys.exit(1)


def feature_extraction(text: str, model: str = "facebook/bart-base"):
    """
    特征提取功能 - 将文本转换为向量表示（embeddings）
    
    参数:
        text: 需要提取特征的文本
        model: 使用的模型，默认为 facebook/bart-base
    
    常用特征提取模型:
        facebook/bart-base - BART 基础模型
        sentence-transformers/all-MiniLM-L6-v2 - 轻量级句子编码器
        sentence-transformers/all-mpnet-base-v2 - 高质量句子编码器
        bert-base-uncased - BERT 基础模型
    
    返回:
        文本的向量表示（numpy array 或 list）
    """
    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
        timeout=30,
    )
    
    print(f"调用模型: {model}")
    print(f"输入文本: {text[:100]}...")  # 只显示前100个字符
    
    result = client.feature_extraction(
        text,
        model=model,
    )
    
    # result 是一个向量（embedding）
    # 通常是多维数组，表示文本的语义特征
    print(f"特征向量维度: {len(result) if hasattr(result, '__len__') else 'unknown'}")
    
    return result


if __name__ == "__main__":
    # 测试代码
    test_texts = [
        "Today is a sunny day and I will get some ice cream.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
    ]
    
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("使用测试文本...")
        user_input = test_texts[0]
    
    if user_input:
        try:
            features = feature_extraction(user_input)
            print(f"\n原文本: {user_input}")
            print(f"特征向量类型: {type(features)}")
            
            # 尝试显示向量的形状和前几个值
            if hasattr(features, '__len__'):
                print(f"向量长度: {len(features)}")
                if hasattr(features[0], '__len__'):
                    print(f"向量形状: {len(features)} x {len(features[0])}")
                    print(f"前3个值: {features[0][:3]}")
                else:
                    print(f"前10个值: {features[:10]}")
            else:
                print(f"特征: {features}")
                
        except Exception as e:
            print(f"\n特征提取失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有输入文本！")
        print("用法: python src/FeatureExtraction.py '文本内容'")
