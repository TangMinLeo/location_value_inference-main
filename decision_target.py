import argparse
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv

from llm_model import server_llm_invoke

VALUE_DIMENSIONS = [
    ("quiet", "安静"),
    ("curiosity", "好奇"),
    ("social", "社交"),
    ("routine", "守规律"),
    ("entertainment", "娱乐"),
    ("health", "健康"),
    ("family", "家庭"),
    ("responsibility", "责任"),
]

PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_scalar(value):
    if value is None:
        return "未知"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

def normalize_values(raw_values):
    values = {}
    for key, _ in VALUE_DIMENSIONS:
        try:
            number = float(raw_values.get(key, 0.0))
        except (TypeError, ValueError):
            number = 0.0
        values[key] = max(-1.0, min(1.0, number))
    return values

def render_values(values):
    ordered = sorted(VALUE_DIMENSIONS, key=lambda item: abs(values[item[0]]), reverse=True)
    lines = []
    for rank, (key, chinese) in enumerate(ordered, start=1):
        lines.append(f"- {key}({chinese}): {values[key]:+.2f}（影响强度第{rank}）")
    return "\n".join(lines)

def read_area_table(path):
    table = pd.read_csv(path, encoding="utf-8-sig")
    table.columns = [column.strip() for column in table.columns]
    return table

def locate_seat_row(table, target):
    floor = str(target.get("floor", "")).strip()
    box = str(target.get("box", "")).strip()
    if not box or box == "nan":
        return None
    rows = table[table["对应看台包厢号"].astype(str).str.strip() == box]
    if floor:
        rows = rows[rows["UE楼层"].astype(str).str.strip() == floor]
    if len(rows) == 0:
        return None
    return rows.iloc[0]

def build_destinations(table, seat_row, seat_id, seat_floor):
    destinations = []
    if seat_row is not None:
        destinations.append({
            "id": seat_id,
            "name": f"我的座位({seat_row['对应看台包厢号']})",
            "desc": f"{seat_row['UE楼层']} {seat_row['ABCD区']}区 {seat_row['功能名称']}",
        })

    public_rows = table[table["对应看台包厢号"].isna()]
    records = []
    for _, row in public_rows.iterrows():
        name = str(row["功能名称"]).strip()
        if not name or name == "nan":
            continue
        records.append({
            "id": str(int(row["ID"])),
            "name": name,
            "floor": str(row["UE楼层"]).strip(),
            "desc": f"{row['UE楼层']} {row['ABCD区']}区",
        })

    records.sort(key=lambda item: (item["floor"] != seat_floor, item["name"]))
    seen = set()
    for record in records:
        signature = (record["floor"], record["name"])
        if signature in seen:
            continue
        seen.add(signature)
        destinations.append({
            "id": record["id"],
            "name": record["name"],
            "desc": record["desc"],
        })
    return destinations

def render_destinations(destinations):
    return "\n".join(
        f"- {item['id']} | {item['name']} | {item['desc']}" for item in destinations
    )

def build_prompt(template, context):
    prompt = template
    for key, value in context.items():
        prompt = re.sub(
            r"\{\{\s*" + key + r"\s*\}\}",
            lambda _match, text=str(value): text,
            prompt,
        )
    remaining = sorted(set(PLACEHOLDER_PATTERN.findall(prompt)))
    if remaining:
        print(f"[警告] 模板中存在未填充的占位符: {remaining}")
    return prompt

def parse_decision(text):
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"模型输出中未找到 JSON: {text}")
    return json.loads(cleaned[start:end + 1])

def repair_decision(decision, valid_ids, seat_id):
    kept = []
    for item in decision.get("destinations") or []:
        location_id = str(item.get("location_id", "")).strip()
        if location_id in valid_ids:
            kept.append({
                "location_id": location_id,
                "reason": str(item.get("reason", "")),
            })
        else:
            print(f"[警告] 丢弃不在候选列表中的目的地: {location_id}")
    if not kept or kept[-1]["location_id"] != seat_id:
        kept.append({
            "location_id": seat_id,
            "reason": "入场阶段最终必须到达我的座位",
        })
    decision["destinations"] = kept
    decision["rationale"] = str(decision.get("rationale", ""))
    return decision

def resolve_target(person, agent_id, target_file):
    target = person.get("target")
    if target:
        return target
    if os.path.exists(target_file):
        target_persons = read_json(target_file)
        return (target_persons.get(agent_id) or {}).get("target") or {}
    return {}


def extract_destination_ids(prompt):
    ids = set()
    for line in prompt.splitlines():
        match = re.match(r"^\s*-\s*(\w+)\s*\|", line)
        if match:
            ids.add(match.group(1))
    return ids


def main():
    parser = argparse.ArgumentParser(description="体育馆演唱会入场阶段目的地决策")
    parser.add_argument("--person_file", default="./person_information_target.json")
    parser.add_argument("--template", default="./prompt.txt")
    # parser.add_argument("--area-csv", default="./功能区域模型标定_精简版.csv")
    parser.add_argument("--agent_id", default=None)
    parser.add_argument("--time_to_show", default="25分钟")
    # parser.add_argument("--current-location", default="2F_C区 出入口门厅（入场口）")
    parser.add_argument("--current_location", default="体育馆外广场")
    parser.add_argument("--current_task", default="入场寻找自己的座位")
    parser.add_argument("--visible_agents", default="较多（入场高峰期，环廊与餐饮区人流密集）")
    parser.add_argument("--output", default='./decision_seq.jsonl')
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    persons = read_json(args.person_file)
    
    # agent_id = args.agent_id or next(iter(persons))
    # if agent_id not in persons:
    #     raise KeyError(f"未找到智能体: {agent_id}")
    # person = persons[agent_id]
    
    count = 0
    
    for agent_id, person in persons.items():
        count += 1
        # target = resolve_target(person, agent_id, args.target_file)
        seat_row = person.get("target", {})
        
        # table = read_area_table(args.area_csv)
        # seat_row = locate_seat_row(table, target)
        time_to_show = person.get("time_to_show", args.time_to_show)
        current_location = person.get("current_location", args.current_location)

        if seat_row is not None:
            seat_id = str(seat_row["box"])
            seat_floor = str(seat_row["floor"]).strip()
            seat_area = str(seat_row["area"]).strip()
            seat_function = str(seat_row["function_name"]).strip()
        else:
            seat_id = str(seat_row.get("box", "")).strip() or "未知座位"
            seat_floor = str(seat_row.get("floor", "未知")).strip()
            seat_area = str(seat_row.get("area", "未知")).strip()
            seat_function = str(seat_row.get("function_name", "观演区域")).strip()

        values = normalize_values(person.get("value", {}))
        
        
        
        # destinations = build_destinations(table, seat_row, seat_id, seat_floor)
        
        # valid_ids = {item["id"] for item in destinations}
        # if seat_id not in valid_ids:
        #     valid_ids.add(seat_id)
        #     destinations.insert(0, {
        #         "id": seat_id,
        #         "name": "我的座位",
        #         "desc": f"{seat_floor} {seat_area}区 {seat_function}",
        #     })

        context = {
            "agent_id": agent_id,
            # "agent_name": str(person.get("name", agent_id)),
            "sub_scenario": "入场",
            "current_location": current_location,
            "time_to_show": time_to_show,
            "current_task": args.current_task,
            "visible_agents": args.visible_agents,
            "age_type": format_scalar(person.get("age")),
            "gender": str(person.get("gender") or "未知"),
            "job": str(person.get("job") or "未知"),
            "seat_id": seat_id,
            "seat_floor": seat_floor,
            "seat_area": seat_area,
            "seat_function": seat_function,
            "value_state": render_values(values),
            # "destination_index": render_destinations(destinations),
        }

        prompt = build_prompt(read_text(args.template), context)
        print("=" * 20 + " 提示词 " + "=" * 20)
        
        if count<6:
            print(prompt)

        if args.dry_run:
            return

        # decision = server_llm_invoke(prompt)


        valid_ids = extract_destination_ids(prompt)
        valid_ids.add(seat_id)
        print(valid_ids)
        
        raw_response = server_llm_invoke(prompt)
        decision = repair_decision(parse_decision(raw_response), valid_ids, seat_id)
        
        
        result = {
            "agent_id": agent_id,
            "sub_scenario": "入场",
            "seat_id": seat_id,
            "destinations": decision["destinations"],
            "rationale": decision["rationale"],
        }

        # print("=" * 20 + " 决策结果 " + "=" * 20)
        # print(json.dumps(result, ensure_ascii=False, indent=2))
        prompt_decision = {
            "prompt": prompt,
            "decision": result,
        }
        if args.output:
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(prompt_decision, ensure_ascii=False) + "\n")
            print(f"结果已保存到: {args.output}")

if __name__ == "__main__":
    main()