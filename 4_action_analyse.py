import pandas as pd
import json

# 读取Excel文件
excel_file = './data/3_address_en.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(excel_file)

# 将数据转换为字典，英文为key，中文为value
category_dict = dict(zip(df['Category_translated'], df['Category']))

# 打印字典
print(category_dict)


# 打开 JSONL 文件
jsonl_file = './data/3_action_gen.jsonl'  # 替换为你的 JSONL 文件路径
action_list = []
# 一行行读取并处理
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():  # 跳过空行
            data = json.loads(line)  # 将 JSON 字符串解析为 Python 字典
            actions=data["action"]
            for action in actions:
                action_list.append(action)
            data["content_ch"]=category_dict[data["category"]]
            jsonl_file_path = 'data/4_action_gen.jsonl'
            with open(jsonl_file_path, 'a', encoding='utf-8') as file:  # 使用 'a' 模式追加写入
                json.dump(data, file, ensure_ascii=False)
                file.write('\n')  # JSONL 文件每行以换行符分隔
            print(data)  # 处理每行数据，如打印
            # 保存到 JSON 文件
            # with open("./data/12_list.json", "w", encoding="utf-8") as file:
            #     json.dump(action_list, file, ensure_ascii=False, indent=4)