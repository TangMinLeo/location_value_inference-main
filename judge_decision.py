import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from tqdm import tqdm
import unicodedata

load_dotenv()

JUDGE_API_URL = os.environ.get(
    "JUDGE_API_URL", "https://api.openai.com/v1/chat/completions"
)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-6-luna")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", '')

# # VALUE_LINE = re.compile(r"^-\s*(\w+)$[^)]*$\s*:\s*([+-]?\d+(?:\.\d+)?)", re.MULTILINE)
# VALUE_LINE = re.compile(r"^-\s*(\w+)\s*$[^)]*$\s*[:：]\s*([+-]?\d+(?:\.\d+)?)", re.MULTILINE)
# CANDIDATE_LINE = re.compile(r"^-\s*(\d+)\s*\|", re.MULTILINE)
# # CITED_VALUE = re.compile(r"(\w+)\s*$\s*([+-]\d+(?:\.\d+)?)\s*$")
# CITED_VALUE = re.compile(r"([A-Za-z_]+)\s*[\(（=:：]\s*([+-]?\d+(?:\.\d+)?)")
# TIME_LINE = re.compile(r"距演出开始\s*[:：]\s*(\d+)")
# SEAT_LINE = re.compile(r"看台包厢号\s*[:：]\s*(\S+)")

# VALUE_LINE = re.compile(r"^-\s*([A-Za-z_]+)\s*$[^)]*$\s*[:：]\s*([+-]?\d+(?:\.\d+)?)", re.MULTILINE)
# CANDIDATE_LINE = re.compile(r"^-\s*(\d+)\s*\|", re.MULTILINE)
# CITED_VALUE = re.compile(r"([A-Za-z_]+)\s*[\(（=:：]\s*([+-]?\d+(?:\.\d+)?)")
# TIME_LINE = re.compile(r"距演出开始\s*[:：]\s*(\d+)")
# SEAT_LINE = re.compile(r"看台包厢号\s*[:：]\s*(\S+)")

VALUE_BLOCK_HEADER = "价值偏好向量"
CANDIDATE_BLOCK_HEADER = "可选目的地"

VALUE_LINE = re.compile(r"^-\s*([A-Za-z_]+)[^0-9+\-]*([+-]?\d+(?:\.\d+)?)", re.MULTILINE)
CANDIDATE_LINE = re.compile(r"^-\s*(\d+)\s*\|", re.MULTILINE)
CITED_VALUE = re.compile(r"([A-Za-z_]+)\s*[\(（=:：]\s*([+-]?\d+(?:\.\d+)?)")
TIME_LINE = re.compile(r"距演出开始\s*[:：]\s*(\d+)")
SEAT_LINE = re.compile(r"看台包厢号\s*[:：]\s*(\S+)")


def normalize_text(text):
    return unicodedata.normalize("NFKC", str(text))


def mentions(text, name):
    return re.search(
        r"(?<![A-Za-z_])" + re.escape(name) + r"(?![A-Za-z_])", text
    ) is not None


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# def read_jsonl(path):
#     records = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 records.append(json.loads(line))
#             except json.JSONDecodeError as exc:
#                 print(f"[警告] 第 {line_no} 行 JSON 解析失败: {exc}")
#     return records


def read_jsonl(path):
    text = read_text(path)
    records = []
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    while pos < length:
        while pos < length and text[pos] in " \t\r\n":
            pos += 1
        if pos >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {text.count(chr(10), 0, pos) + 1} 行附近 JSON 解析失败: {exc}") from exc
        records.append(obj)
        pos = end
    return records





# def parse_value_vector(prompt):
#     values = {}
#     for name, number in VALUE_LINE.findall(prompt):
#         values[name] = float(number)
#     return values


# def parse_value_vector(prompt):
#     values = {}
#     for name, number in VALUE_LINE.findall(normalize_text(prompt)):
#         values[name] = float(number)
#     return values

def parse_value_vector(prompt):
    text = normalize_text(prompt)
    header = text.find(VALUE_BLOCK_HEADER)
    if header == -1:
        raise ValueError("未找到“价值偏好向量”区块，无法解析价值向量")
    start = text.find("\n", header)
    if start == -1:
        raise ValueError("“价值偏好向量”区块格式异常，未找到换行")
    start += 1
    end = text.find(CANDIDATE_BLOCK_HEADER, start)
    if end == -1:
        end = len(text)
    block = text[start:end]
    values = {}
    for name, number in VALUE_LINE.findall(block):
        values.setdefault(name, float(number))
    if not values:
        raise ValueError("价值向量解析为空：区块内未匹配到任何维度，请检查 prompt 格式")
    return values



def parse_candidate_ids(prompt):
    return set(CANDIDATE_LINE.findall(prompt))

def parse_time_to_show(prompt):
    match = TIME_LINE.search(prompt)
    return int(match.group(1)) if match else None

def parse_seat_id(prompt, decision):
    seat = decision.get("seat_id")
    if seat:
        return str(seat).strip()
    match = SEAT_LINE.search(prompt)
    return match.group(1).strip() if match else ""

def build_judge_prompt(template, record, inject_checks, checks):
    prompt = template.replace("{{original_prompt}}", record.get("prompt", ""))
    prompt = prompt.replace(
        "{{decision_json}}",
        json.dumps(record.get("decision", {}), ensure_ascii=False, indent=2),
    )
    if inject_checks:
        block = "\n".join(
            f"- {item['rule']}: {item['result']} | {item['evidence']}" for item in checks
        )
        prompt += (
            "\n\n==================== 九、机器预校验结果 ====================\n\n"
            "以下 J1/J2/J3/J5/J6 已由程序精确计算，结论具备确定性，"
            "你在 checks 中必须与此保持一致，不得推翻：\n\n" + block + "\n"
        )
    return prompt

def machine_checks(record, values, candidates, seat_id, time_to_show):
    decision = record.get("decision") or {}
    dests = decision.get("destinations") or []
    ids = [str(item.get("location_id", "")).strip() for item in dests]
    detour_ids = [item for item in ids if item != seat_id]
    checks = []

    last = ids[-1] if ids else ""
    checks.append({
        "rule": "J1",
        "result": "pass" if last == seat_id else "fail",
        "evidence": f"末项={last or '(空)'}, 座位={seat_id}",
    })

    illegal = [item for item in detour_ids if item not in candidates]
    checks.append({
        "rule": "J2",
        "result": "pass" if not illegal else "fail",
        "evidence": f"非座位目的地={detour_ids or '无'}, 候选列表={sorted(candidates)}, 非法项={illegal or '无'}",
    })

    ok_count = len(detour_ids) <= 2 and len(ids) <= 3
    checks.append({
        "rule": "J3",
        "result": "pass" if ok_count else "fail",
        "evidence": f"绕行数={len(detour_ids)}, 总长度={len(ids)}",
    })

    # rationale = str(decision.get("rationale", ""))
    rationale = normalize_text(decision.get("rationale", ""))
    cited = CITED_VALUE.findall(rationale)
    cited_dims = sorted({name for name, _ in cited})
    named_dims = sorted(name for name in values if mentions(rationale, name))
    checks.append({
        "rule": "J5",
        # "result": "pass" if len(cited_dims) >= 2 else "fail",
        "result": "pass" if len(cited_dims) >= 2 or len(named_dims) >= 2 else "fail",
        # "evidence": f"rationale 引用维度={cited_dims or '无'}（共 {len(cited_dims)} 个）",
        "evidence": f"带数值引用={cited_dims or '无'}（{len(cited_dims)}个）, 仅提及={named_dims or '无'}（{len(named_dims)}个）",
    })

    mismatches = []
    for name, number in cited:
        if name not in values:
            mismatches.append(f"{name}({number}) 不存在于价值向量")
            continue
        if abs(values[name] - float(number)) > 0.02:
            mismatches.append(f"{name} 引用{number} vs 实际{values[name]:+.2f}")
    checks.append({
        "rule": "J6",
        "result": "pass" if not mismatches else "fail",
        "evidence": "; ".join(mismatches) if mismatches else f"全部一致，共校验 {len(cited)} 处",
    })
    
    

    if time_to_show is not None:
        checks.append({
            "rule": "J11",
            "result": "uncertain",
            "evidence": f"距演出={time_to_show}分钟，需结合绕行地点耗时判断（程序不裁决）",
        })
    return checks

def classify(values, ids, seat_id):
    tags = []
    if any(abs(value) >= 0.9 for value in values.values()):
        tags.append("extreme")
    if any(value <= -0.3 for value in values.values()):
        tags.append("negative")
    if len([value for value in values.values() if value > 0.5]) >= 2:
        tags.append("multi_high")
    if any(item != seat_id for item in ids):
        tags.append("detour")
    else:
        tags.append("direct")
    return tags

def sample_records(records, per_group, limit):
    buckets = {}
    prepared = []
    for index, record in enumerate(records):
        decision = record.get("decision") or {}
        values = parse_value_vector(record.get("prompt", ""))
        seat_id = parse_seat_id(record.get("prompt", ""), decision)
        ids = [
            str(item.get("location_id", "")).strip()
            for item in (decision.get("destinations") or [])
        ]
        tags = classify(values, ids, seat_id)
        prepared.append({"index": index, "tags": tags})

    for item in prepared:
        for tag in item["tags"]:
            buckets.setdefault(tag, [])
            if len(buckets[tag]) < per_group:
                buckets[tag].append(item["index"])

    selected = []
    seen = set()
    for tag, indexes in buckets.items():
        for index in indexes:
            if index in seen:
                continue
            seen.add(index)
            selected.append(index)
    selected.sort()
    if limit and len(selected) > limit:
        selected = selected[:limit]
    return selected

# def call_judge(prompt, args):
#     headers = {"Content-Type": "application/json"}
#     if JUDGE_API_KEY:
#         headers["Authorization"] = f"Bearer {JUDGE_API_KEY}"
#     payload = {
#         "model": JUDGE_MODEL,
#         "messages": [
#             {"role": "system", "content": "你是严格的决策审查员，只输出 JSON。"},
#             {"role": "user", "content": prompt},
#         ],
#         "temperature": args.temperature,
#     }
#     if args.max_tokens:
#         payload["max_tokens"] = args.max_tokens

#     for attempt in range(1, args.retries + 1):
#         try:
#             resp = requests.post(
#                 JUDGE_API_URL, json=payload, headers=headers, timeout=args.timeout
#             )
#             if resp.status_code == 429 or resp.status_code >= 500:
#                 raise requests.exceptions.RequestException(
#                     f"HTTP {resp.status_code}: {resp.text[:200]}"
#                 )
#             resp.raise_for_status()
#             result = resp.json()
#             return result["choices"][0]["message"]["content"] or ""
#         except requests.exceptions.RequestException as exc:
#             if attempt >= args.retries:
#                 raise
#             wait = min(30, 5 * attempt)
#             print(f"[重试] 第{attempt}/{args.retries}次失败: {exc}，{wait}秒后重试")
#             time.sleep(wait)
#     raise RuntimeError("裁判模型调用失败")


NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}

def call_judge(prompt, args):
    headers = {"Content-Type": "application/json"}
    if JUDGE_API_KEY:
        headers["Authorization"] = f"Bearer {JUDGE_API_KEY}"

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "你是严格的决策审查员，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.max_tokens:
        payload["max_completion_tokens"] = args.max_tokens

    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            resp = requests.post(
                JUDGE_API_URL, json=payload, headers=headers, timeout=args.timeout
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"网络异常: {exc}"
            if attempt >= args.retries:
                raise RuntimeError(last_error)
            wait = min(30, 5 * attempt)
            print(f"[重试] 第{attempt}/{args.retries}次网络失败: {exc}，{wait}秒后重试")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            result = resp.json()
            return result["choices"][0]["message"]["content"] or ""

        body = resp.text[:800]
        if resp.status_code in NON_RETRYABLE_STATUS:
            raise RuntimeError(
                f"HTTP {resp.status_code} 请求被拒绝（不重试），响应体: {body}\n"
                f"当前请求体: {json.dumps(payload, ensure_ascii=False)[:300]}"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code}: {body}"
            if attempt >= args.retries:
                raise RuntimeError(last_error)
            wait = min(30, 5 * attempt)
            print(f"[重试] 第{attempt}/{args.retries}次限流/服务端错误，{wait}秒后重试")
            time.sleep(wait)
            continue

        raise RuntimeError(f"HTTP {resp.status_code} 未知错误: {body}")

    raise RuntimeError(f"裁判模型调用失败: {last_error}")


def parse_judge_output(text):
    if not text or not text.strip():
        raise ValueError("裁判模型返回空内容")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"裁判输出中未找到 JSON: {text[:200]}")
    return json.loads(cleaned[start:end + 1])

def judge_one(index, record, template, args):
    prompt_text = record.get("prompt", "")
    decision = record.get("decision") or {}
    values = parse_value_vector(prompt_text)
    if not values:
        print(f"[警告] 智能体 {decision.get('agent_id')} 的价值向量解析为空，请检查 prompt 格式")
    
    candidates = parse_candidate_ids(prompt_text)
    seat_id = parse_seat_id(prompt_text, decision)
    time_to_show = parse_time_to_show(prompt_text)

    checks = machine_checks(record, values, candidates, seat_id, time_to_show)
    judge_prompt = build_judge_prompt(template, record, args.inject_machine_checks, checks)

    item = {
        "index": index,
        "agent_id": decision.get("agent_id") or record.get("agent_id", ""),
        "seat_id": seat_id,
        "time_to_show": f'{time_to_show}分钟',
        "value_vector": values,
        "machine_checks": checks,
        "judge": None,
        "judge_raw": "",
        "error": None,
    }

    if args.dry_run:
        item["judge_prompt"] = judge_prompt
        return item

    try:
        raw = call_judge(judge_prompt, args)
        item["judge_raw"] = raw
        item["judge"] = parse_judge_output(raw)
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item

def write_result(handle, item):
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()

def aggregate(results):
    judged = [item for item in results if item.get("judge")]
    failed = [item for item in results if not item.get("judge")]

    scores = [item["judge"].get("score") for item in judged if isinstance(item["judge"].get("score"), (int, float))]
    verdicts = Counter(item["judge"].get("verdict", "未知") for item in judged)

    issue_types = Counter()
    defect_counter = Counter()
    for item in judged:
        for issue in item["judge"].get("issues") or []:
            issue_types[issue.get("type", "未知")] += 1
        for defect in item["judge"].get("template_defects") or []:
            defect_counter[str(defect)[:2]] += 1

    rule_stats = {}
    for item in results:
        for check in item.get("machine_checks") or []:
            stats = rule_stats.setdefault(check["rule"], Counter())
            stats[check["result"]] += 1

    summary = {
        "total": len(results),
        "judged": len(judged),
        "failed": len(failed),
        "score_mean": round(statistics.mean(scores), 2) if scores else None,
        "score_median": statistics.median(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "verdict_distribution": dict(verdicts),
        "issue_type_distribution": dict(issue_types),
        "template_defect_hits": dict(defect_counter),
        "machine_check_stats": {rule: dict(stats) for rule, stats in rule_stats.items()},
        "failed_items": [
            {"index": item["index"], "agent_id": item.get("agent_id"), "error": item.get("error")}
            for item in failed
        ],
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="体育馆入场决策批量裁判")
    parser.add_argument("--input", default="./decision_seq.jsonl")
    parser.add_argument("--template", default="./judge_prompt_template.txt")
    parser.add_argument("--output", default="./judge_results.jsonl")
    parser.add_argument("--summary", default="./judge_summary.json")
    parser.add_argument("--workers", type=int, default=1, help="并发数，1 表示串行")
    parser.add_argument("--per-group", type=int, default=0, help="每个分层标签抽样条数，0 表示不抽样")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inject-machine-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    template = read_text(args.template)
    records = read_jsonl(args.input)
    if not records:
        print(f"[错误] 输入文件为空: {args.input}")
        return

    if args.per_group > 0:
        indexes = sample_records(records, args.per_group, args.limit)
        print(f"[抽样] 共 {len(records)} 条，分层抽样后 {len(indexes)} 条")
    else:
        indexes = list(range(len(records)))
        if args.limit:
            indexes = indexes[: args.limit]

    done = set()
    if args.resume and os.path.exists(args.output):
        for item in read_jsonl(args.output):
            if item.get("judge"):
                done.add(item.get("index"))
        print(f"[续跑] 已完成 {len(done)} 条，跳过")
    indexes = [index for index in indexes if index not in done]

    if not indexes:
        print("[提示] 无待处理记录")
        return

    if args.dry_run:
        item = judge_one(indexes[0], records[indexes[0]], template, args)
        print("=" * 20 + " 裁判提示词预览 " + "=" * 20)
        print(item["judge_prompt"])
        print("=" * 20 + " 机器预校验 " + "=" * 20)
        print(json.dumps(item["machine_checks"], ensure_ascii=False, indent=2))
        return

    results = []
    mode = "串行" if args.workers <= 1 else f"并发({args.workers}线程)"
    print(f"[模式] {mode}，待裁判 {len(indexes)} 条，裁判模型={JUDGE_MODEL}")

    with open(args.output, "a", encoding="utf-8") as handle:
        if args.workers <= 1:
            for index in tqdm(indexes, desc="裁判进度"):
                item = judge_one(index, records[index], template, args)
                write_result(handle, item)
                results.append(item)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(judge_one, index, records[index], template, args): index
                    for index in indexes
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="裁判进度"):
                    try:
                        item = future.result()
                    except Exception as exc:
                        index = futures[future]
                        item = {"index": index, "judge": None, "error": str(exc)}
                    write_result(handle, item)
                    results.append(item)

    summary = aggregate(results)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)

    print("=" * 20 + " 汇总统计 " + "=" * 20)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已写入: {args.output}")
    print(f"汇总已写入: {args.summary}")

if __name__ == "__main__":
    main()