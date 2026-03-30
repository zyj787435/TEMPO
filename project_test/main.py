#!/usr/bin/env python3
import os
import sys
import json
import httpx
from openai import OpenAI
from pathlib import Path
from datetime import datetime

# 导入所有 HuggingFace 模型函数
from src.TextGeneration import textGeneration
from src.Summarization import summarization
from src.QuestionAnswering import question_answering
from src.Translation import translation
from src.TexttoImage import text_to_image
from src.FeatureExtraction import feature_extraction

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    print("请先在环境变量中设置 OPENAI_API_KEY（或使用 .env 文件并加载）。")
    sys.exit(1)
    
# 创建 OpenAI 客户端对象
client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(
        timeout=30.0
    )
)


def load_model_info():
    """加载模型信息配置文件"""
    info_path = Path(__file__).parent / "info.json"
    # __file__是当前程序的绝对路径
    # Path是一个类，功能丰富，其重载了运算符 /，可以拼接路径
    with open(info_path, 'r', encoding='utf-8') as f:
    # with 是上下文管理器，不用手动调用close()，打开出错自动关闭文件等等
        return json.load(f)
    # 返回json格式的file内容

def get_user_input():
    """获取用户输入"""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    print("请输入你的任务描述（结束后按 Enter）：")
    return sys.stdin.readline().strip()


def select_model_with_gpt(task_description: str, model_info: dict):
    """
    使用 ChatGPT 根据任务描述选择合适的模型并生成提示词
    
    返回格式: {
        "task_type": "TextGeneration" | "Summarization" | "QuestionAnswering" | "Translation" | "TextToImage" | "FeatureExtraction" | "NONE",
        "prompt": "生成的提示词",
        "additional_params": {}  # 额外参数，如翻译的源语言和目标语言
    }
    """
    # 构造给 ChatGPT 的 prompt
    system_prompt = f"""你是一个AI任务分类器。用户会描述一个任务，你需要从以下6种模型中选择最合适的一个，并生成相应的提示词。

    可用模型：
    {json.dumps(model_info, indent=2, ensure_ascii=False)}

    请严格按照以下JSON格式返回，不要添加任何其他文字：
    {{
        "task_type": "模型类型名称（TextGeneration/Summarization/QuestionAnswering/Translation/TextToImage/FeatureExtraction）或 NONE",
        "prompt": "处理后的提示词或输入文本",
        "additional_params": {{}}
    }}

    规则：
    1. 如果任务描述明确属于某个模型的功能范围，选择该模型
    2. 对于QuestionAnswering，需要同时提供question和context，其中context在additional_params中设置，这里如果用户输入没有提供上下文context，你需要按照用户的意思自动生成非空的context作为你的返回
    3. 对于Translation，需要在additional_params中指定src_lang和tgt_lang（使用mBART-50格式，如en_XX, zh_CN）
    4. 如果没有合适的模型，返回task_type为"NONE"
    5. 只返回JSON，不要有任何额外文字或解释"""

    # 在f-string中，{{和}}用来转义{和}
    # json.dumps用于将Python字典转换为JSON字符串
    # json.loads用于将JSON字符串转换为Python字典
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"任务描述：{task_description}"}
            ],
            max_tokens=10000,
            temperature=0.3,  # 降低随机性，使选择更稳定
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"\n[ChatGPT 分析结果]")
        print(result_text)
        print()
        
        # 解析 JSON 返回
        result = json.loads(result_text)
        return result
        
    except json.JSONDecodeError as e:
        print(f"解析 ChatGPT 返回的 JSON 失败: {e}")
        print(f"原始返回: {result_text}")
        return {"task_type": "NONE", "prompt": "", "additional_params": {}}
    except Exception as e:
        print(f"调用 ChatGPT 出错：{type(e).__name__}: {e}")
        return {"task_type": "NONE", "prompt": "", "additional_params": {}}


def execute_task(task_type: str, prompt: str, additional_params: dict, model_info: dict):
    """执行选定的任务"""
    
    if task_type == "NONE":
        print("ChatGPT 判断该任务无法由现有模型完成。")
        return None
    
    if task_type not in model_info:
        print(f"未知的任务类型: {task_type}")
        return None
    
    model_name = model_info[task_type]["model"]
    print(f"任务类型: {task_type}")
    print(f"使用模型: {model_name}")
    print(f"提示词: {prompt}")
    print()
    
    try:
        result = None
        
        if task_type == "TextGeneration":
            result = textGeneration(prompt, model_name)
            
        elif task_type == "Summarization":
            result = summarization(prompt, model_name)
            
        elif task_type == "QuestionAnswering":
            question = additional_params.get("question", prompt)
            context = additional_params.get("context", "")
            if not context:
                print("问答任务需要提供上下文（context）")
                return None
            result = question_answering(question, context, model_name)
            
        elif task_type == "Translation":
            src_lang = additional_params.get("src_lang", "en_XX")
            tgt_lang = additional_params.get("tgt_lang", "zh_CN")
            result = translation(prompt, src_lang, tgt_lang, model_name)
            
        elif task_type == "TextToImage":
            # 图片生成返回的是文件路径
            result = text_to_image(prompt, model_name)
            
        elif task_type == "FeatureExtraction":
            result = feature_extraction(prompt, model_name)
            # 特征向量太长，只保存不打印
            print(f"特征提取完成，向量维度: {len(result)}")
            
        return result
        
    except Exception as e:
        print(f"执行任务时出错：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_result(task_type: str, user_input: str, result: any):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 根据任务类型选择保存方式
    if task_type == "TextToImage":
        # 图片已经保存，result 是文件路径
        pass      
    elif task_type == "FeatureExtraction":
        # 保存特征向量到文件
        output_file = f"feature_vector_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "input": user_input,
                "features": result,
                "timestamp": timestamp
            }, f, indent=2, ensure_ascii=False)
        print(f"\n特征向量已保存到: {output_file}")
        
    else:
        # 文本结果保存到文件
        output_file = f"result_{task_type}_{timestamp}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"任务类型: {task_type}\n")
            f.write(f"用户输入: {user_input}\n")
            f.write(f"时间: {timestamp}\n")
            f.write("-" * 50 + "\n")
            f.write(f"结果:\n{result}\n")
        print(f"结果已保存到: {output_file}")


def main():
    """主函数"""
    
    # 1. 获取用户输入
    user_input = get_user_input()
    if not user_input:
        print("没有输入，退出。")
        return
    
    print(f"\n用户任务: {user_input}\n")
    
    # 2. 加载模型信息
    model_info = load_model_info()
    
    # 3. 使用 ChatGPT 选择模型并生成提示词
    print("正在分析任务并选择模型...")
    selection = select_model_with_gpt(user_input, model_info)
    
    task_type = selection.get("task_type")
    prompt = selection.get("prompt", "")
    additional_params = selection.get("additional_params", {})
    
    # 4. 执行任务
    print("正在执行任务...\n")
    result = execute_task(task_type, prompt, additional_params, model_info)
    
    # 5. 显示和保存结果
    if result is not None:
        if task_type not in ["TextToImage", "FeatureExtraction"]:
            print("\n" + "=" * 60)
            print("执行结果:")
            print("=" * 60)
            print(result)
        
        save_result(task_type, user_input, result)
        print("\n任务完成！")
    else:
        print("\n任务执行失败。")


if __name__ == "__main__":
    main()