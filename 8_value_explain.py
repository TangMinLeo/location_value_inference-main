import os
import json
import asyncio
from dotenv import load_dotenv
import aiohttp  # 确保添加了aiohttp依赖

# 加载环境变量
load_dotenv()
# BASE_URL = os.environ.get("BASE_URL", "http://172.16.40.183:8000/v1/chat/completions")
BASE_URL = os.environ.get("BASE_URL", "http://172.16.41.254:30245/v1/chat/completions")


# ---------------------- LLM 调用工具函数 ----------------------
async def request_llm(url, headers, data):
    """异步请求LLM服务"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        print(f"LLM请求失败: {str(e)}")
        return None

async def generate_explanation(prompt):
    """调用本地LLM生成价值观解释"""
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {
        # "model": "llama3.1-8b-chinese-chat",
        "model": "/root/kube-user/Qwen3.8-27B",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "seed": 0
    }
    result = await request_llm(BASE_URL, headers, data)

    if not result:
        return None
    try:
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, json.JSONDecodeError) as e:
        print(f"响应解析失败: {str(e)}，原始响应: {result}")
        return None

# ---------------------- 核心业务逻辑 ----------------------
SYSTEM_PROMPT = """[Background]
# Values are the intrinsic goods or ideals that people pursue or cherish, such as happiness, well-being, justice, or freedom.
# Values are the desirable qualities that people may seek in their lives and in the world.
# They are often the guiding principles for individuals and societies, shaping goals, motivations, and preferences.
# ---
#
# [Task]
# Provide a brief explanation based on the given values, explaining their meaning in 15 words or less.
# Output in JSON format.
# ---
#
# [Example]
# **value:** "integrity"
# Your response: { "integrity": "Acting with honesty and uprightness, always adhering to moral principles and commitments." }
# **value:** "responsibility"
# Your response: { "responsibility": "Being accountable for one's actions and their impact, and actively fulfilling obligations." }
# **value:** "integrity"
# Your response: { "trust": "Holding a sense of confidence and security in the actions, words, and intentions of others." }
# ---
"""  # 保持原有系统提示不变

USER_PROMPT_TEMPLATE = "**value:** {text}"


def process_values(input_path="data/7_3_all_values.json", output_path="data/8_all_values_explain.json"):
    """主流程：读取数据 -> 生成解释 -> 保存结果（含进度显示）"""
    with open(input_path, 'r', encoding='utf-8') as f:
        values_data = json.load(f)

    total = len(values_data)
    print(f"▶ 开始处理价值观解释生成，共 {total} 个需要处理...")

    results = {}
    for idx, (value, _) in enumerate(values_data.items(), 1):
        progress_percent = (idx / total) * 100
        try:
            user_prompt = USER_PROMPT_TEMPLATE.format(text=value)
            explanation = asyncio.run(generate_explanation(user_prompt))

            if explanation and isinstance(explanation, dict) and value in explanation:
                results[value] = explanation[value]
                print(f"[成功] [{progress_percent:.2f}%] ({idx}/{total}) {value} ➜ {explanation[value]}")
            else:
                print(f"[跳过] [{progress_percent:.2f}%] ({idx}/{total}) {value}：未获取到有效解释")
        except Exception as e:
            print(f"[错误] [{progress_percent:.2f}%] ({idx}/{total}) {value} ➜ 错误原因: {str(e)}")
            continue

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n✔ 处理完成！成功处理 {len(results)}/{total} 个价值观，结果保存至: {output_path}")

if __name__ == "__main__":
    process_values(input_path="data/7_2_filtered_values_final.json", output_path="data/8_all_values_explain.json")
    # process_values()
    



# import pandas as pd
# import json
# import json
# from openai import OpenAI
# import os
# import pandas as pd
#
# # Set your OpenAI API key
# os.environ[
#     'OPENAI_API_KEY'] = ""
# client = OpenAI()
#
# SYSTEM_PROMPT = """[Background]
# Values are the intrinsic goods or ideals that people pursue or cherish, such as happiness, well-being, justice, or freedom.
# Values are the desirable qualities that people may seek in their lives and in the world.
# They are often the guiding principles for individuals and societies, shaping goals, motivations, and preferences.
# ---
#
# [Task]
# Provide a brief explanation based on the given values, explaining their meaning in 15 words or less.
# Output in JSON format.
# ---
#
# [Example]
# **value:** "integrity"
# Your response: { "integrity": "Acting with honesty and uprightness, always adhering to moral principles and commitments." }
# **value:** "responsibility"
# Your response: { "responsibility": "Being accountable for one's actions and their impact, and actively fulfilling obligations." }
# **value:** "integrity"
# Your response: { "trust": "Holding a sense of confidence and security in the actions, words, and intentions of others." }
# ---
# """
#
# USER_PROMPT_TEMPLATE = "**value:** {text}"
#
#
# def get_openai_response(prompt):
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": prompt}
#         ],
#         response_format={"type": "json_object"}
#     )
#     message = completion.choices[0].message.content
#     return message
#
#
# if __name__ == '__main__':
#     # 打开并读取JSON文件
#     with open('./data/7_3_all_values.json', 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     i=0
#     for key, value in data.items():
#         i=i+1
#         try:
#             # 生成OpenAI的输入
#             prompt = USER_PROMPT_TEMPLATE.format(text=key)
#             # 调用OpenAI的API
#             response = get_openai_response(prompt)
#             # 将OpenAI的输出添加到数据中
#             response=json.loads(response)
#             data[key] = response[key]
#             print(i)
#         except Exception as e:
#             print(i)
#             print(e)
#             continue
#     # 打印读取的数据
#     with open("data/8_all_values_explain.json", 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
#     print(data)