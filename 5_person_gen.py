import json
import os
import pandas as pd
import asyncio
import aiohttp
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
# url0 = f'http://172.16.40.183:8000/v1/chat/completions'  # 保持原自定义服务地址

url0 = "http://172.16.41.254:30245/v1/chat/completions"  # 本地模型接口地址
# MODEL_NAME = "/root/kube-user/Qwen3.8-27B"  # 模型名称

async def request_url(url, headers, data):
    """异步请求函数"""
    result = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                assert response.status == 200
                result = await response.json()
    except Exception as e:
        print(f"请求失败: {str(e)}")
    return result

async def generate_response(message):
    """生成LLM响应"""
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "/root/kube-user/Qwen3.8-27B",  # 可根据需要修改模型
        'messages': [{"role": "user", "content": message}],
        "temperature": 0,
        "seed": 0,
    }
    result = await request_url(url0, headers, data)
    # 响应格式校验
    if not isinstance(result, dict):
        print(f"无效响应类型: {type(result)}，预期字典")
        return "{}"
    choices = result.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        print(f"无效choices字段: {choices}")
        return "{}"
    content = choices[0].get("message", {}).get("content", "{}")
    return content if isinstance(content, str) else "{}"

def llm_invoke(text):
    """同步调用LLM接口"""
    print(f"输入内容: {text}")
    result = asyncio.run(generate_response(text))
    print(f"输出内容: {result}")
    return result

def main():
    data_list = []
    input_file_path = "./data/4_action_gen.jsonl"

    # 读取输入数据
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line.strip()))

    # 处理数据（跳过前289条）
    for idx, row in enumerate(data_list, 1):
        # if idx <= 289:
        #     continue

        # 重构action字段
        actions = row['action']
        row['action'] = {}
        row['action_list'] = actions

        # 遍历每个action生成结果
        for ac in actions:
            prompt = f'''
                For the given action, provide the subject of the action and generate a sentence that includes the subject. 
                If multiple subjects are involved, provide different subjects; if only one subject is involved, provide only that subject. 
                Output in JSON format
                example1:
                action: consulting financial advisors
                output: {{ "consultant" : "As a consultant, seek financial advice on financial matters.","financial advisor": "As a financial advisor, provide financial management advice to others."}}
                example2:
                action: cooking delicious food
                output: {{ "cook":"As a cook, prepare delicious food."}}
                example3:
                action: attending school events together
                output:{{"child": "As a child, attend school events with parents.","parent": "As a parent, accompany children to school events."}}
                The given action is as follows:
                action: {ac}
                Output:
            '''
            # 调用自定义LLM获取响应
            response = llm_invoke(prompt)

            # 解析JSON（添加异常处理）
            try:
                row['action'][ac] = {"common": json.loads(response)}
            except json.JSONDecodeError:
                print(f"JSON解析失败，原始响应: {response}")
                row['action'][ac] = {"common": {}}

        # 写入输出文件
        jsonl_file_path = './data/5_person_gen.jsonl'
        with open(jsonl_file_path, 'a', encoding='utf-8') as file:
            json.dump(row, file, ensure_ascii=False)
            file.write('\n')

        print(f"处理进度: {idx}/{len(data_list)}")


if __name__ == '__main__':
    main()

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
# def get_openai_response(prompt):
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         response_format={"type": "json_object"}
#     )
#     message = completion.choices[0].message.content
#     return message
#
#
# if __name__ == '__main__':
#     data_list = []
#     input_file_path="./data/4_action_gen.jsonl"
#     with open(input_file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 将每一行解析为字典并加入列表
#             data_list.append(json.loads(line.strip()))
#     i=0
#     # Iterate over each row in the DataFrame
#     for row in data_list:
#         i=i+1
#         if i<=289:
#             continue
#         actions = row['action']
#         row['action']={}
#         row['action_list']=actions
#
#
#
#         for ac in actions:
#             prompt = f'''
#                        For the given action, provide the subject of the action and generate a sentence that includes the subject. If multiple subjects are involved, provide different subjects; if only one subject is involved, provide only that subject. Output in JSON format
#                        example1:
#                        action: consulting financial advisors
#                        output: {{ "consultant" : "As a consultant, seek financial advice on financial matters.","financial advisor": "As a financial advisor, provide financial management advice to others."}}
#                        example2:
#                        action: cooking delicious food
#                        output: {{ "cook":"As a cook, prepare delicious food."}}
#                        example3:
#                        action: attending school events together
#                        output:{{"child": "As a child, attend school events with parents.","parent": "As a parent, accompany children to school events."}}
#                        The given action is as follows:
#                        action: {ac}
#                        Output:
#                        '''
#             row['action'][ac] = {}
#             row['action'][ac]["common"] = json.loads(get_openai_response(prompt))
#             # print(prompt)
#
#         jsonl_file_path= './data/5_person_gen.jsonl'
#         with open(jsonl_file_path, 'a', encoding='utf-8') as file:  # 使用 'a' 模式追加写入
#             json.dump(row, file, ensure_ascii=False)
#             file.write('\n')  # JSONL 文件每行以换行符分隔
#         print(i)
#
#
#
#
