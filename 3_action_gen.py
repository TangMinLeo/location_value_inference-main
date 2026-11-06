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

# LLM_MODEL = "llama3.1-8b-chinese-chat"  # 可根据需要修改模型名称
# url0 = f'http://172.16.40.183:8000/v1/chat/completions'  # 保持原接口地址

url0 = "http://172.16.41.254:30245/v1/chat/completions"  # 本地模型接口地址
LLM_MODEL = "/root/kube-user/Qwen3.8-27B"  # 模型名称


# ---------------------- LLM 交互相关函数 ----------------------
async def request_url(url, headers, data):
    """异步请求接口"""
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
    """生成模型响应"""
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "model": LLM_MODEL,
        'messages': [
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    return await request_url(url0, headers, data)


def llm_invoke(text):
    """同步调用LLM接口"""
    print(f"输入内容: {text}")
    result = asyncio.run(generate_response(text))

    # 响应校验逻辑
    if not isinstance(result, dict):
        print(f"无效响应类型: {type(result)}")
        return "{}"

    choices = result.get("choices", [])
    if not choices or not isinstance(choices[0].get("message"), dict):
        print("未获取有效choices或message字段")
        return "{}"

    content = choices[0]["message"].get("content", "{}")
    print(f"输出内容: {content}")
    return content


# ---------------------- 核心业务逻辑 ----------------------
def get_llm_response(prompt):
    """调用自定义LLM获取响应"""
    response = llm_invoke(prompt)
    # 确保返回JSON字符串（可根据模型实际输出调整）
    # return response if response.startswith("{") else "{}"
    import re
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```', '', response)
    json_match = re.search(r'\{.*\}', response.strip(), flags=re.DOTALL)
    if json_match:
        return json_match.group()
    return "{}"


def main():
    # 读取Excel文件
    file_path = './data/3_address_en.xlsx'
    df = pd.read_excel(file_path, sheet_name=0)

    # # 提取去重后的分类列表
    # unique_categories = df['Category'].dropna().unique().tolist()
    # unique_translated = df['Category_translated'].dropna().unique().tolist()
    # print(f"总分类数: {len(unique_categories)}")
    # print(f"总翻译分类数: {len(unique_translated)}")
    
    # 建立中文→英文的映射字典（基于原始行数据，保证配对正确）
    category_map = dict(zip(df['Category'], df['Category_translated']))
    categories = list(category_map.keys())
    translated = list(category_map.values())
    print(f"总分类数: {len(categories)}")
    print(f"去重后英文翻译数: {len(set(translated))}")
    
    unique_translated = list(set(translated))
    
    from collections import Counter
    trans_counts = Counter(translated)
    duplicates = {k: v for k, v in trans_counts.items() if v > 1}
    if duplicates:
        print(f"以下英文翻译对应多个中文: {duplicates}")

    inverse_map = {v: k for k, v in category_map.items()}
    for item in duplicates.keys():    
        print(item, inverse_map[item])
        
    
    assert len(categories) == len(translated), "原始分类与翻译分类数量不一致"
    print(f"原始分类示例: {categories[:5]}")
    print(f"翻译分类示例: {translated[:5]}")

    # 生成动作并保存结果
    results = []
    jsonl_path = 'data/3_action_gen.jsonl'

    for i in range(len(unique_translated)):
        # original = unique_categories[i]
        translated = unique_translated[i]
        original = inverse_map[translated]

        # 构造提示（与原示例格式一致）
        prompt = f'''请生成与给定内容相关的动作。要求动作不要过于具体，保持一致的具体程度。每个动作应并列列出，无层级关系。输出JSON格式。
示例1:
Content: hotel
Output:
{{"category": "hotel", "action": ["staying at the hotel", "dining at the hotel", "enjoying entertainment activities", "providing hotel services to others"]}}
示例2:
Content: tourist attractions
Output:
{{"category": "tourist attractions", "action": ["shopping", "sightseeing", "eating delicious food"]}}

Input: {translated}
Output:'''

        # 调用LLM获取响应并解析
        output = get_llm_response(prompt)
        try:
            parsed = json.loads(output)
            parsed["content_original"] = original  # 添加原始分类字段
            results.append(parsed)

            # 写入JSONL文件
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                json.dump(parsed, f, ensure_ascii=False)
                f.write('\n')

        except json.JSONDecodeError:
            print(f"第{i}条响应解析失败: {output}")
            continue

    # 保存到Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel('./data/3_action_gen.xlsx', index=False)
    print("处理完成，结果已保存")


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
#
#
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
#
# if __name__ == '__main__':
#     # 读取 Excel 文件
#     file_path = './data/3_address_en.xlsx'  # 替换为你的文件路径
#     sheet_name = 0  # 如果需要指定表格名称或索引，修改此处
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#
#     # 提取 'first_category_name_translated' 列并去重
#     unique_list = df['Category'].dropna().unique().tolist()
#     unique_list_translated = df['Category_translated'].dropna().unique().tolist()
#
#     # 输出结果
#     print(unique_list)
#     print(unique_list_translated)
#
#
#     results = []
#     # Create a list to store results
#     i=0
#     for i in range(0, len(unique_list)):
#         content=unique_list[i]
#         content_translated=unique_list_translated[i]
#         prompt = '''Please generate actions related to the given content. Ensure that the actions provided are not overly specific and maintain consistent levels of specificity. Each action should be listed in parallel, without hierarchical relationships. Output in JSON format..
# Example1:
# Content: hotel
# Output:
# {
# 'category': 'hotel',
# 'action': ['staying at the hotel', 'dining at the hotel', 'enjoying entertainment activities', 'providing hotel services to others']
# }
# Example2:
# Content: tourist attractions
# Output:
# {
# 'category': 'tourist attractions',
# 'action': ['shopping', 'sightseeing', 'eating delicious food']
# }
#
# Input: ''' + content_translated+"\n"+"output:"
#         output = get_openai_response(prompt)
#         parsed_output = json.loads(output)
#         parsed_output["content_ch"]=content
#         jsonl_file_path= 'data/3_action_gen.jsonl'
#         with open(jsonl_file_path, 'a', encoding='utf-8') as file:  # 使用 'a' 模式追加写入
#             json.dump(parsed_output, file, ensure_ascii=False)
#             file.write('\n')  # JSONL 文件每行以换行符分隔
#
#         results.append(parsed_output)
#         print(i)
#
#     output_file = './data/3_action_gen.xlsx'
#     # Save results to a new Excel file
#     results_df = pd.DataFrame(results)
#     results_df.to_excel(output_file, index=False)
#
