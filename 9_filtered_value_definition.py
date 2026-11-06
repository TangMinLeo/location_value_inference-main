import os

from rouge_score import rouge_scorer
import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer  # 本地嵌入模型库
import aiohttp
import asyncio
from llm_model import server_llm_invoke
from dotenv import load_dotenv

# ---------------------- 本地大模型调用配置 ----------------------
async def request_url(url, headers, data):
    """异步请求本地大模型API"""
    result = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                assert response.status == 200
                result = await response.json()
    except Exception as e:
        print(f"请求失败: {e}")
    return result

async def generate_response(message):
    """生成大模型响应"""
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "llama3.1-8b-chinese-chat",  # 本地部署的模型名称
        "messages": [{"role": "user", "content": message}],
        "temperature": 0,
        "seed": 0,
    }
    url = "http://172.16.40.183:8000/v1/chat/completions"  # 本地大模型API地址
    return await request_url(url, headers, data)




def llm_invoke(text):
    """调用本地大模型解析/翻译文本"""
    print(f"输入内容: {text}")
    result = asyncio.run(generate_response(text))

    # 解析响应（根据本地模型API返回格式调整）
    if not result or "choices" not in result:
        print("无效响应")
        return ""
    content = result["choices"][0].get("message", {}).get("content", "")
    print(f"输出内容: {content}")
    return content

# ---------------------- 文本嵌入与去重逻辑 ----------------------
def cosine_similarity(a, b):
    """余弦相似度计算（不变）"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_sentence_embedding(input_texts: list[str]):
    """使用本地嵌入模型生成向量（替换原OpenAI接口）"""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 加载本地嵌入模型（可替换为其他模型）
    # model = SentenceTransformer('models/sentence-transformers--all-MiniLM-L6-v2/snapshots/master')  # 加载本地嵌入模型（可替换为其他模型）
    embeddings = model.encode(input_texts)  # 直接生成向量
    # 保存嵌入结果（与原逻辑一致）
    file_path = "./data/9_0_embeddings_order_new.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings.tolist()  # 转换为列表格式

def get_embeddings(values):
    """获取文本嵌入（调用本地方法）"""
    return get_sentence_embedding(values)

def deduplicate_rouge(values, rouge_type='rougeL', threshold=0.25):
    """ROUGE去重（逻辑不变）"""
    threshold = {'fmeasure': threshold, 'precision': threshold, 'recall': threshold}
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    idx2drop = []
    for i in range(1, len(values)):
        print(f"ROUGE去重处理中... {i}/{len(values)}")
        for j in range(i):
            if j in idx2drop:
                continue
            value_i, value_j = values[i], values[j]
            scores = scorer.score(value_i, value_j)[rouge_type]
            if (scores.fmeasure > threshold['fmeasure'] or
                    scores.precision > threshold['precision'] or
                    scores.recall > threshold['recall']):
                idx2drop.append(i)
                break
    return [value for idx, value in enumerate(values) if idx not in idx2drop]


def deduplicate_embed(values, threshold=0.53):
    """嵌入向量去重（逻辑不变）"""
    embeddings = get_embeddings(values)
    idx2drop = []
    for i in range(1, len(values)):
        print(f"嵌入去重处理中... {i}/{len(values)}")
        for j in range(i):
            if j not in idx2drop:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim > threshold:
                    idx2drop.append(i)
                    break
    return [value for idx, value in enumerate(values) if idx not in idx2drop]


# ---------------------- 主流程（融合解析/翻译） ----------------------
if __name__ == '__main__':
    load_dotenv()
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_ENDPOINT'] = 'https://modelscope.cn'
    # 1. 读取原始数据
    with open('./data/8_all_values_explain.json', 'r', encoding='utf-8') as file:
        value_dict = json.load(file)

    # 2. 使用本地大模型解析/翻译原始内容（示例：翻译为中文）
    # processed_values = []
    # for key, value in value_dict.items():
    #     # 调用本地大模型翻译（根据需求调整提示词）
    #     # translated_value = llm_invoke(f"将以下内容翻译为中文：{value}")
    #     translated_value = server_llm_invoke(f"将以下内容翻译为中文：{value}")
    #     processed_values.append(translated_value)
    #     print(f"原始内容: {value} → 翻译后: {translated_value}")
    processed_values = list(value_dict.values())

    # 3. 去重处理
    filtered_by_embed = deduplicate_embed(processed_values)
    print(f"嵌入去重后数量: {len(filtered_by_embed)}")

    filtered_by_rouge = deduplicate_rouge(filtered_by_embed)
    print(f"ROUGE去重后数量: {len(filtered_by_rouge)}")

    # 4. 保存结果
    final_values = {k: v for k, v in value_dict.items() if v in filtered_by_rouge}
    with open("data/9_3_filtered_values_0.25_order_new.json", 'w', encoding='utf-8') as f:
        json.dump(final_values, f, ensure_ascii=False, indent=4)




    value_dict = final_values
    # 2. 使用本地大模型解析/翻译，带断点缓存（中断后缓存结果无需重新请求）
    cache_file = "./data/9_cache_translated2_new.json"
    os.makedirs("./data", exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"[INFO] 加载缓存，已有 {len(cache)} 条记录")
    else:
        cache = {}
    processed_values = []
    for key, value in value_dict.items():
        if key in cache:
            translated_value = cache[key]
            print(f"[缓存命中] {key}: {value} → {translated_value}")
        else:
            try:
                translated_value = server_llm_invoke(f"将以下内容翻译为中文：{value}")
                if not translated_value:
                    translated_value = value  # 模型返回空时用原文兜底
            except Exception as e:
                print(f"[调用失败] {key}: {e}，使用原文兜底")
                translated_value = value
            cache[key] = translated_value
            # 每翻译一条立即刷新缓存，确保中断不丢失
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"[模型调用] {value} → {translated_value}")
        processed_values.append(translated_value)



    # 4. 保存结果
    final_values = dict(zip(value_dict.keys(), processed_values))
    with open("data/9_3_filtered_values_0.25_order_translated_new.json", 'w', encoding='utf-8') as f:
        json.dump(final_values, f, ensure_ascii=False, indent=4)


# from rouge_score import rouge_scorer
# import json
# from openai import OpenAI
# import os
# import numpy as np
# import pickle
#
#
# # Set your OpenAI API key
# os.environ[
#     'OPENAI_API_KEY'] = ""
# client = OpenAI()
#
#
# def cosine_similarity(a, b):
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
#
#
#
# def get_openai_sentence_embedding(input_texts: list[str], model_name: str='text-embedding-3-large'):
#     """
#     Get the sentence embeddings of the input texts using OpenAI API.
#
#     Args:
#         input_texts (list[str]): A list of input texts.
#         model_name (str): The name of the model.
#         api_key (str): The API key of OpenAI.
#     """
#     from openai import OpenAI
#     client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#     response = client.embeddings.create(
#         input=input_texts,
#         model=model_name,
#     )
#
#     embeddings = [response.data[i].embedding for i in range(len(response.data))]
#     file_path="./data/9_0_embeddings_order.pkl"
#     with open(file_path, 'wb') as f:
#         pickle.dump(embeddings, f)
#     # with open(file_path, 'rb') as f:
#     #     return pickle.load(f)
#     return embeddings
#
#
# def get_embeddings(values):
#     embeddings=get_openai_sentence_embedding(values, 'text-embedding-3-large')
#     return embeddings
#
#
# def deduplicate_rouge(values, rouge_type='rougeL', threshold=0.25):
#     threshold = {'fmeasure': threshold, 'precision': threshold, 'recall': threshold}
#
#     scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
#
#     idx2drop = []
#     for i in range(1, len(values)): # lower priority
#         print("rouge",i)
#         for j in range(i): # higher priority
#             if j in idx2drop: # skip if already marked for removal
#                 continue
#             value_i = values[i]
#             value_j = values[j]
#             scores = scorer.score(value_i, value_j)[rouge_type]
#             if scores.fmeasure > threshold['fmeasure'] or scores.precision > threshold['precision'] or scores.recall > threshold['recall']:
#                 idx2drop.append(i)
#                 break
#
#     filtered_values = [value for idx, value in enumerate(values) if idx not in idx2drop]
#     return filtered_values
#
#
# def deduplicate_embed(values, threshold=0.53, relevance_threshold=None, dedup_across_vrd=False, force_embeds=False):
#
#     embeddings = get_embeddings(values)
#     idx2drop = []
#     for i in range(1, len(values)): # lower priority
#         print("embed", i)
#         for j in range(i):
#             if j not in idx2drop:
#                 sim = cosine_similarity(embeddings[i], embeddings[j])
#                 if sim > threshold:
#                     idx2drop.append(i)
#                     break
#
#     filtered_values = [value for idx, value in enumerate(values) if idx not in idx2drop]
#     return filtered_values
#
#
# if __name__ == '__main__':
#
#     with open('./data/8_1_all_values_explain_order.json', 'r', encoding='utf-8') as file:
#         value_dict = json.load(file)
#
#     values=[]
#     for key, value in value_dict.items():
#         values.append(value)
#     print(len(values))
#     # # Deduplicate using ROUGE
#     # filtered_values = deduplicate_rouge(values)
#     # print(len(filtered_values))
#     # # Deduplicate using embeddings
#     # filtered_values = deduplicate_embed(filtered_values)
#     # print(len(filtered_values))
#
#     filtered_values = deduplicate_embed(values)
#     print(len(filtered_values))
#     filtered_values = deduplicate_rouge(filtered_values)
#     print(len(filtered_values))
#
#
#
#     final_values={k: v for k, v in value_dict.items() if v in filtered_values}
#     with open("data/9_3_filtered_values_0.25_order.json", 'w', encoding='utf-8') as f:
#         json.dump(final_values, f, ensure_ascii=False, indent=4)
#
#     # 0.25 105
#     # 0.2  55
