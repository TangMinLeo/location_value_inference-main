
# `1_address_gen.py` | `data/1_ods_bd_poi.csv` | `data/2_address.xlsx` | 从原始 POI 数据中提取一级/二级/三级分类并去重，得到结构化的地点分类表。
python3 1_address_gen.py

# data/2_address.xlsx	data/3_address_en.xlsx	调用 LLM 将中文地点分类翻译成英文，为后续生成动作做准备。两个脚本功能重复，任选其一即可。
python3 2_address.py

# data/3_address_en.xlsx	data/3_action_gen.jsonl、data/3_action_gen.xlsx	针对每个英文地点分类，生成人们在相应地点可能进行的相关动作（action）。
python3 3_action_gen.py

# data/3_action_gen.jsonl、data/3_address_en.xlsx	data/4_action_gen.jsonl	将英文分类映射回中文，并整理出结构化的动作列表，建立"地点 → 动作"的映射。
python3 4_action_analyse.py

# data/4_action_gen.jsonl	data/5_person_gen.jsonl	对每个动作生成"主体（person）"及描述该主体的例句，建立"动作 → 主体"的映射。
python3 5_person_gen.py

# data/5_person_gen.jsonl	data/6_value_gen.jsonl	根据主体例句推理出与该主体相关的价值观（values）及解释，建立"主体 → 价值观"的映射。
python3 6_value_gen.py

# data/6_value_gen.jsonl	data/7_2_filtered_values_final.json	翻译价值观解释，并基于 sentence embedding 与 ROUGE 指标对价值观进行去重，得到初步筛选后的价值观集合。
python3 7_filter_value.py

# data/7_3_all_values.json	data/8_all_values_explain.json  8_1_all_values_explain_order	为每个价值观重新生成规范的英文解释。
python3 8_value_explain.py

# data/8_1_all_values_explain_order.json	data/9_3_filtered_values_0.25_order.json	将英文解释翻译为中文，并再次基于 embedding + ROUGE 去重，输出最终的中文价值观词典。
python3 9_filtered_value_definition.py