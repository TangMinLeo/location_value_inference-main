# 地点价值观推断流水线

本项目是一套基于大语言模型（LLM）的离线批处理脚本集合，目标是从百度 POI（兴趣点）分类数据出发，逐层推理出"地点 → 动作 → 主体 → 价值观"的层级关系，最终得到一份结构化的中文价值观词典及其解释。

## 核心输入与输出

### 核心输入

- `data/1_ods_bd_poi.csv`
  - 原始百度 POI 分类数据，包含各级地点分类信息。

### 核心输出

- `data/9_3_filtered_values_0.25_order.json`
  - 最终去重、排序后的中文价值观词典，包含每个价值观的中文定义与解释。
- `data/11_all_value_dict.json`、`data/12_all_action_list.json`、`data/13_all_person_list.json`、`data/14_all_value_list.json`、`data/15_all_person_list.json`
  - 各阶段的聚合统计文件，用于汇总动作、主体、价值观等要素的频次与分布。

## 运行方式

所有脚本均应在项目根目录下按编号顺序执行：

```bash
cd /home/pku/python_project/location_value_inference
python3 1_address_gen.py
python3 2_address.py          # 2_address.py 与 2_address_translate.py 功能重复，任选其一即可
python3 3_action_gen.py
python3 4_action_analyse.py
python3 5_person_gen.py
python3 6_value_gen.py
python3 7_filter_value.py
python3 8_value_explain.py
python3 9_filtered_value_definition.py
```

运行前请确认：

1. 本地 LLM 服务 `http://172.16.40.183:8000/v1/chat completions` 可访问。
2. 已安装必要依赖：`pandas`、`openpyxl`、`aiohttp`、`python-dotenv`、`numpy`、`tqdm`、`rouge-score`、`sentence-transformers`。
3. 脚本重复执行前建议清理对应的 `.jsonl` 输出文件，避免以追加模式写入导致数据重复。

## 各脚本及中间数据说明

| 步骤 | 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 1 | `1_address_gen.py` | `data/1_ods_bd_poi.csv` | `data/2_address.xlsx` | 从原始 POI 数据中提取一级/二级/三级分类并去重，得到结构化的地点分类表。 |
| 2 | `2_address.py` / `2_address_translate.py` | `data/2_address.xlsx` | `data/3_address_en.xlsx` | 调用 LLM 将中文地点分类翻译成英文，为后续生成动作做准备。两个脚本功能重复，任选其一即可。 |
| 3 | `3_action_gen.py` | `data/3_address_en.xlsx` | `data/3_action_gen.jsonl`、`data/3_action_gen.xlsx` | 针对每个英文地点分类，生成人们在相应地点可能进行的相关动作（action）。 |
| 4 | `4_action_analyse.py` | `data/3_action_gen.jsonl`、`data/3_address_en.xlsx` | `data/4_action_gen.jsonl` | 将英文分类映射回中文，并整理出结构化的动作列表，建立"地点 → 动作"的映射。 |
| 5 | `5_person_gen.py` | `data/4_action_gen.jsonl` | `data/5_person_gen.jsonl` | 对每个动作生成"主体（person）"及描述该主体的例句，建立"动作 → 主体"的映射。 |
| 6 | `6_value_gen.py` | `data/5_person_gen.jsonl` | `data/6_value_gen.jsonl` | 根据主体例句推理出与该主体相关的价值观（values）及解释，建立"主体 → 价值观"的映射。 |
| 7 | `7_filter_value.py` | `data/6_value_gen.jsonl` | `data/7_2_filtered_values_final.json` | 翻译价值观解释，并基于 sentence embedding 与 ROUGE 指标对价值观进行去重，得到初步筛选后的价值观集合。 |
| 8 | `8_value_explain.py` | `data/7_3_all_values.json` | `data/8_all_values_explain.json` | 为每个价值观重新生成规范的英文解释。注意：当前脚本中的系统提示为占位符，使用前需补全。 |
| 9 | `9_filtered_value_definition.py` | `data/8_1_all_values_explain_order.json` | `data/9_3_filtered_values_0.25_order.json` | 将英文解释翻译为中文，并再次基于 embedding + ROUGE 去重，输出最终的中文价值观词典。 |


