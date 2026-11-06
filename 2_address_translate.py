import pandas as pd
import os
import asyncio  # 新增异步支持库
import aiohttp  # 新增HTTP客户端库

# 读取Excel文件（原始逻辑保留）
file_path = './data/2_address.xlsx'
df = pd.read_excel(file_path)

# 配置自定义大模型参数（根据实际接口修改）
MODEL_API_URL = "http://172.16.40.183:8000/v1/chat/completions"  # 本地模型接口地址
MODEL_NAME = "llama3.1-8b-chinese-chat"  # 模型名称
HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

async def request_model(url, headers, data):
    """异步请求模型接口（封装通用请求逻辑）"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()  # 检查HTTP状态码
                return await response.json()  # 返回JSON格式响应
    except Exception as e:
        print(f"模型请求失败：{str(e)}")
        return None


async def generate_translation(text):
    """调用自定义模型生成翻译（适配模型输入输出格式）"""
    # 构造请求体（与您的模型API要求一致）
    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user",
            "content": f"Only output the translated content, and the first letters are all lowercase. Translate this to English: {text}"
             }
        ],
        "temperature": 0,
        "seed": 0
    }

    # 调用模型接口
    response = await request_model(MODEL_API_URL, HEADERS, request_data)

    # 解析响应（根据实际返回结构调整）
    if not response:
        return text  # 请求失败时返回原文

    # 校验响应结构（关键字段检查）
    choices = response.get("choices", [])
    if not choices or not isinstance(choices[0].get("message"), dict):
        print(f"无效响应结构：{response}")
        return text

    translated_content = choices[0]["message"].get("content", "")
    return translated_content.strip() or text  # 若内容为空则返回原文


def translate_text(text):
    """同步入口函数（供pandas.apply调用）"""
    # 异步函数需要在同步上下文中运行
    return asyncio.run(generate_translation(text))


# 翻译Excel列（原始逻辑保留，替换翻译函数）
df['Category_translated'] = df['Category'].apply(translate_text)

# 保存翻译结果（原始逻辑保留）
output_path = './data/3_address_en.xlsx'
df.to_excel(output_path, index=False)

print(f"翻译完成，结果已保存到 '{output_path}'")



# import pandas as pd
# import openai  # 必须安装openai库
# import os
# from openai import OpenAI
#
#
#
# # 读取Excel文件
# file_path = './data/2_address.xlsx'  # 替换为您的Excel文件路径
# df = pd.read_excel(file_path)
#
#
# os.environ[
#     'OPENAI_API_KEY'] = ""
# client = OpenAI()
#
#
# def translate_text(text):
#     prompt = f"Only output the translated content, and the first letters are all lowercase. Translate this to English: {text}"
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     message = completion.choices[0].message.content
#     print(message)
#     return message
#
#
# # 翻译每一列的内容，并保存到新的列中
# df['Category_translated'] = df['Category'].apply(translate_text)
#
#
# # 保存翻译后的Excel文件
# df.to_excel('./data/3_address_en.xlsx', index=False)
#
# print("翻译完成，结果已保存到 'translated_excel_file.xlsx'")
