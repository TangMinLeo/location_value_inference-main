import os
import json
import pickle
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm
from rouge_score import rouge_scorer
from dotenv import load_dotenv

# ------------------ 本地大模型调用配置 ------------------
load_dotenv()
key = os.environ.get("API_KEY")
url = os.environ.get("BASE_URL")
# url0 = f"http://172.16.40.183:8000/v1/chat/completions"
url0 = "http://172.16.41.254:30245/v1/chat/completions"  # 本地模型接口地址


async def request_url(url, headers, data):
    result = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                assert response.status == 200
                result = await response.json()
    except Exception as e:
        print(e)
    return result


async def generate_response(message):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "/root/kube-user/Qwen3.8-27B",
        "messages": [
            {"content": message, "role": "user"},
        ],
        "temperature": 0,
        "seed": 0,
    }
    result = await request_url(url0, headers, data)
    if not isinstance(result, dict):
        print(f"无效响应类型：{type(result)}，预期字典")
        return ""
    choices = result.get("choices", [])
    if not isinstance(choices, list) or len(choices) == 0:
        print(f"无效 choices 字段：{choices}，预期非空列表")
        return ""
    message_obj = choices[0].get("message", {})
    content = message_obj.get("content", "")
    if not isinstance(content, str):
        print(f"无效 content 字段：{content}，预期字符串")
        return ""
    return content


def llm_invoke(text):
    """调用本地大模型（同步封装）"""
    print(f"input: {text}")
    result = asyncio.run(generate_response(text))
    print(f"output: {result}")
    return result


# ------------------ 向量处理和去重 ------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_random_embeddings(values, dim=768):
    """模拟生成随机向量（避免依赖外部嵌入服务）"""
    print(f"[模拟] 生成{len(values)}条嵌入向量")
    return [np.random.rand(dim) for _ in values]


def deduplicate_by_embedding(values, threshold=0.53):
    embeddings = generate_random_embeddings(values)
    keep = []
    for i in tqdm(range(len(values)), desc="基于嵌入去重"):
        emb_i = embeddings[i]
        duplicate = any(cosine_similarity(emb_i, embeddings[j]) > threshold for j in keep)
        if not duplicate:
            keep.append(i)
    return [values[i] for i in keep]


def deduplicate_by_rouge(values, rouge_type='rougeL', threshold=0.2):
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    keep = []
    for i in tqdm(range(len(values)), desc="基于ROUGE去重"):
        text_i = values[i]
        if all(scorer.score(text_i, values[j])[rouge_type].fmeasure < threshold for j in keep):
            keep.append(i)
    return [values[i] for i in keep]


# ------------------ 主逻辑 ------------------

if __name__ == "__main__":
    input_file_path = "./data/6_value_gen.jsonl"
    output_file_path = "./data/7_2_filtered_values_final.json"

    os.makedirs("./data", exist_ok=True)

    # 读取数据
    data_list = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line.strip()))

    values = []
    value_dict = {}

    # 提取数据并翻译
    for row in tqdm(data_list, desc="解析数据中"):
        person_dict = row['action']
        for kaction, vaction in person_dict.items():
            vaction_common = vaction["common"]
            for kperson, vperson in vaction_common.items():
                value_list = vperson["values"]
                for value in value_list:
                    v_name = value["value"]
                    explanation = value["explanation"]
                    if v_name not in value_dict:
                        # translated = llm_invoke(f"请将以下中文解释翻译成英文：{explanation}")
                        # translated = llm_invoke(f"请将以下解释翻译成英文：{explanation}")
                        value_dict[v_name] = {
                            "original_explanation": explanation,
                            # "translated_explanation": translated
                        }
                        values.append(explanation)

    print(f"\n[INFO] 原始数量: {len(values)}")
    print(values[0])
    
    # 去重处理
    # values = deduplicate_by_embedding(values)
    # print(f"[INFO] 嵌入去重后数量: {len(values)}")

    # values = deduplicate_by_rouge(values)
    # print(f"[INFO] ROUGE去重后数量: {len(values)}")
    # print(values[:10])

    # 过滤最终结果
    # final_values = {k: v for k, v in value_dict.items() if v["original_explanation"] in values}
    # final_values = {k: v["original_explanation"] for k, v in value_dict.items() if v["original_explanation"] in values}
    final_values = {k: v["original_explanation"] for k, v in value_dict.items()}
    
    

    # 保存
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_values, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 处理完成，最终保存到 {output_file_path}")




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
# def get_openai_sentence_embedding(input_texts: list[str], model_name: str='text-embedding-3-large'):
#     # """
#     # Get the sentence embeddings of the input texts using OpenAI API.
#     #
#     # Args:
#     #     input_texts (list[str]): A list of input texts.
#     #     model_name (str): The name of the model.
#     #     api_key (str): The API key of OpenAI.
#     # """
#     # from openai import OpenAI
#     # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#     # response = client.embeddings.create(
#     #     input=input_texts,
#     #     model=model_name,
#     # )
#     #
#     # embeddings = [response.data[i].embedding for i in range(len(response.data))]
#     file_path="./data/7_1_embeddings.pkl"
#     # with open(file_path, 'wb') as f:
#     #     pickle.dump(embeddings, f)
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)
#     # return embeddings
#
#
# def get_embeddings(values):
#     embeddings=get_openai_sentence_embedding(values, 'text-embedding-3-large')
#     return embeddings
#
#
# def deduplicate_rouge(values, rouge_type='rougeL', threshold=0.2):
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
#     data_list = []
#     input_file_path="./data/6_value_gen.jsonl"
#     with open(input_file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 将每一行解析为字典并加入列表
#             data_list.append(json.loads(line.strip()))
#
#     values=[]
#     values_value=[]
#     value_dict={}
#     # Iterate over each row in the DataFrame
#     for row in data_list:
#         person_dict=row['action']
#         for kaction, vaction in person_dict.items():
#             vaction = vaction["common"]
#             for kperson, vperson in vaction.items():
#                 value_list=vperson["values"]
#                 for value in value_list:
#
#                     v=value["value"]
#                     k=value["explanation"]
#                     # 如果v不在values_value中
#                     if v not in values_value:
#                         value_dict[v]=k
#                         # 解释
#                         values.append(k)
#                         # 价值观
#                         values_value.append(v)
#
#     with open("data/7_3_all_values.json", 'w', encoding='utf-8') as f:
#         json.dump(value_dict, f, ensure_ascii=False, indent=4)
#
#
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
#     with open("data/7_2_filtered_values_0.2.json", 'w', encoding='utf-8') as f:
#         json.dump(final_values, f, ensure_ascii=False, indent=4)
