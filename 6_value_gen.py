import pandas as pd
import json
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
# url0 = f'http://172.16.40.183:8000/v1/chat/completions'  # 保持原自定义服务地址
url0 = "http://172.16.41.254:30245/v1/chat/completions"  # 本地模型接口地址

# 系统提示词保持不变
SYSTEM_PROMPT = """[Background]
Values are the intrinsic goods or ideals that people pursue or cherish, such as happiness, well-being, justice, or freedom.
Values are the desirable qualities that people may seek in their lives and in the world.
They are often the guiding principles for individuals and societies, shaping goals, motivations, and preferences.
---

[Task]
Reason about the person's relevant values based on the given perception.
Please generate any and all potentially relevant values that the person may hold, even if they sometimes oppose each other.
With each generated value, give explanation of what the value means given the context within 15 words.
Output in JSON format.
---

[Example]
    **Perception:** "Interest to be in charge and tell others what to do."
Your response: { "values": [{"value": "power", "explanation": "Social status and prestige, control or dominance over people and resources."}, {"value":"autonomy", "explanation": "independence, self-governance, and the ability to make personal decisions"} ] }
---
"""

USER_PROMPT_TEMPLATE = "**Perception:** {text}"


async def request_url(url, headers, data):
    """异步请求函数（保留原实现）"""
    result = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                assert response.status == 200
                result = await response.json()
    except Exception as e:
        print(f"请求错误: {e}")
    return result


async def generate_response(message):
    """生成LLM响应（保留原实现）"""
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "/root/kube-user/Qwen3.8-27B",
        'messages': [
            {"role": "system", "content": SYSTEM_PROMPT},  # 整合系统提示
            {"role": "user", "content": message}
        ],
        "temperature": 0,
        "seed": 0,
    }
    return await request_url(url0, headers, data)


def llm_invoke(text):
    """调用LLM获取响应（修改后版本）"""
    print(f"输入内容: {text}")
    result = asyncio.run(generate_response(text))

    # 响应验证（保留原逻辑）
    if not isinstance(result, dict):
        print(f"无效响应类型: {type(result)}，预期字典")
        return ""

    choices = result.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        print(f"无效choices字段: {choices}，预期非空列表")
        return ""

    first_choice = choices[0]
    message_obj = first_choice.get("message", {})
    content = message_obj.get("content", "")

    if not isinstance(content, str):
        print(f"无效content字段: {content}，预期字符串")
        return ""

    print(f"输出内容: {content}")
    return content


def process_data():
    """主处理逻辑（整合原6_value_gen.py）"""
    input_file_path = "./data/5_person_gen.jsonl"
    jsonl_file_path = './data/6_value_gen.jsonl'

    with open(input_file_path, 'r', encoding='utf-8') as file:
        data_list = [json.loads(line.strip()) for line in file]

    for idx, row in enumerate(data_list, 1):
        person_dict = row['action']
        for kaction, vaction in person_dict.items():
            vaction = vaction["common"]
            for kperson, vperson in vaction.items():
                # 构造用户提示
                user_prompt = USER_PROMPT_TEMPLATE.format(text=vperson)
                # 调用自定义LLM获取响应
                output = llm_invoke(user_prompt)

                # 解析并存储结果
                try:
                    output_json = json.loads(output)
                    row['action'][kaction]["common"][kperson] = {
                        "person_action": vperson,
                        "values": output_json.get("values", [])
                    }
                except json.JSONDecodeError:
                    print(f"第{idx}条记录JSON解析失败，内容: {output}")
                    row['action'][kaction]["common"][kperson] = {
                        "person_action": vperson,
                        "values": []
                    }

        # 写入结果文件
        with open(jsonl_file_path, 'a', encoding='utf-8') as file:
            json.dump(row, file, ensure_ascii=False)
            file.write('\n')

        print(f"处理进度: {idx}/{len(data_list)}")


if __name__ == '__main__':
    process_data()

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
# Reason about the person's relevant values based on the given perception.
# Please generate any and all potentially relevant values that the person may hold, even if they sometimes oppose each other.
# With each generated value, give explanation of what the value means given the context within 15 words.
# Output in JSON format.
# ---
#
# [Example]
#     **Perception:** "Interest to be in charge and tell others what to do."
# Your response: { "values": [{"value": "power", "explanation": "Social status and prestige, control or dominance over people and resources."}, {"value":"autonomy", "explanation": "independence, self-governance, and the ability to make personal decisions"} ] }
# ---
# """
#
# USER_PROMPT_TEMPLATE = "**Perception:** {text}"
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
#
#     data_list = []
#     input_file_path="./data/5_person_gen.jsonl"
#     with open(input_file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 将每一行解析为字典并加入列表
#             data_list.append(json.loads(line.strip()))
#     i=0
#     # Iterate over each row in the DataFrame
#     for row in data_list:
#         i=i+1
#
#         person_dict=row['action']
#         for kaction, vaction in person_dict.items():
#             vaction = vaction["common"]
#             for kperson, vperson in vaction.items():
#
#
#                 prompt = user_prompt = USER_PROMPT_TEMPLATE.format(text=vperson)
#                 # print(prompt)
#                 output=get_openai_response(prompt)
#                 # print(output)
#                 output=json.loads(output)
#                 row['action'][kaction]["common"][kperson]={}
#                 row['action'][kaction]["common"][kperson]["person_action"] = vperson
#                 row['action'][kaction]["common"][kperson]["values"]=output["values"]
#
#         jsonl_file_path = './data/6_value_gen.jsonl'
#         with open(jsonl_file_path, 'a', encoding='utf-8') as file:  # 使用 'a' 模式追加写入
#             json.dump(row, file, ensure_ascii=False)
#             file.write('\n')  # JSONL 文件每行以换行符分隔
#
#
#         print(i)
#
#
