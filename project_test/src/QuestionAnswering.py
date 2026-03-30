#!/usr/bin/env python3

import os
import sys
from huggingface_hub import InferenceClient

token = os.getenv("HUGGINGFACE_TOKEN")

if not token:
    print("环境变量 HUGGINGFACE_TOKEN 未设置！")
    sys.exit(1)


def question_answering(question: str, context: str, model: str = "deepset/roberta-base-squad2"):
    """
    问答功能 - 根据上下文回答问题
    
    参数:
        question: 要问的问题
        context: 包含答案的上下文文本
        model: 使用的模型，默认为 deepset/roberta-base-squad2
    
    常用问答模型:
        deepset/roberta-base-squad2 - 基于 SQuAD 2.0 训练的 RoBERTa
        distilbert-base-cased-distilled-squad - 轻量级 DistilBERT
        bert-large-uncased-whole-word-masking-finetuned-squad - BERT Large
    
    返回:
        答案文本
    """
    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
        timeout=30,
    )
    
    print(f"调用模型: {model}")
    #print(f"问题: {question}")
    
    result = client.question_answering(
        question=question,
        context=context,
        model=model,
    )
    
    # result 包含答案和置信度分数
    if hasattr(result, 'answer'):
        answer = result.answer
        score = getattr(result, 'score', None)
        if score:
            print(f"置信度: {score:.4f}")
    else:
        answer = str(result)
    
    return answer


if __name__ == "__main__":
    # 测试代码
    test_cases = [
        {
            "question": "What is my name?",
            "context": "My name is Clara and I live in Berkeley.",
        },
        {
            "question": "Where do I live?",
            "context": "My name is Clara and I live in Berkeley.",
        },
        {
            "question": "What is the capital of France?",
            "context": "Paris is the capital and most populous city of France. It is located in the north-central part of the country.",
        },
    ]
    
    if len(sys.argv) > 2:
        # 命令行参数: question context
        question = sys.argv[1]
        context = " ".join(sys.argv[2:])
    else:
        print("使用测试用例...")
        test_case = test_cases[0]
        question = test_case["question"]
        context = test_case["context"]
    
    if question and context:
        answer = question_answering(question, context)
        print(f"\n上下文: {context}")
        print(f"问题: {question}")
        print(f"答案: {answer}")
    else:
        print("缺少问题或上下文！")
        print("用法: python src/QuestionAnswering.py '问题' '上下文文本'")
