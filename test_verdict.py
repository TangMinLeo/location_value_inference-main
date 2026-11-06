import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"

def resolve_base_url():
    url = os.environ.get("JUDGE_API_URL", "").strip()
    if not url:
        return DEFAULT_BASE_URL
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")].rstrip("/")
    return url.rstrip("/")

BASE_URL = resolve_base_url()
API_KEY = os.environ.get("JUDGE_API_KEY", "")
MODEL = os.environ.get("JUDGE_MODEL", "")

DIAGNOSIS = [
    ("model_not_found", "模型 ID 不存在：用 --list-models 查真实 ID"),
    ("does not exist", "模型 ID 不存在：检查是否漏写日期快照后缀"),
    ("unsupported value", "参数取值不被接受：多为 temperature 只能为 1，去掉 --temperature"),
    ("temperature", "该模型对 temperature 有限制：尝试不传该参数"),
    ("max_tokens", "该模型改用 max_completion_tokens：去掉 --max-tokens"),
    ("invalid_api_key", "密钥无效：检查 .env 是否被读取、是否已吊销"),
    ("incorrect api key", "密钥无效：确认无引号、无多余空格"),
    ("insufficient_quota", "额度不足：检查账户余额或改用其他模型"),
    ("context_length", "输入超长：减少提示词长度"),
    ("content_filter", "内容被安全策略拦截：检查提示词是否含敏感描述"),
]

def mask_key(key):
    if not key:
        return "(空)"
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"

def build_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers

def print_config():
    print("=" * 62)
    print("当前配置")
    print(f"  BASE_URL : {BASE_URL}")
    print(f"  MODEL    : {MODEL or '(未设置，请设 JUDGE_MODEL)'}")
    print(f"  API_KEY  : {mask_key(API_KEY)}")
    print("=" * 62)
    if not API_KEY:
        print("[警告] JUDGE_API_KEY 为空，请求必然返回 401")

def diagnose(body):
    text = body.lower()
    hits = [hint for keyword, hint in DIAGNOSIS if keyword in text]
    if hits:
        print("[诊断建议]")
        for hint in hits:
            print(f"  - {hint}")
    else:
        print("[诊断建议] 未匹配到已知错误特征，请直接查看上方响应体")

def list_models(keyword):
    url = f"{BASE_URL}/models"
    print(f"\n[1/2] 拉取模型列表: GET {url}")
    try:
        resp = requests.get(url, headers=build_headers(), timeout=30)
    except requests.exceptions.RequestException as exc:
        print(f"[失败] 网络异常: {exc}")
        return False

    if resp.status_code != 200:
        print(f"[失败] HTTP {resp.status_code}")
        print(f"响应体: {resp.text[:800]}")
        diagnose(resp.text)
        return False

    data = resp.json().get("data") or []
    print(f"[成功] 共 {len(data)} 个模型")
    matched = [item.get("id", "") for item in data if keyword.lower() in item.get("id", "").lower()]
    if matched:
        print(f"\n包含 '{keyword}' 的模型（请原样填入 JUDGE_MODEL）:")
        for model_id in matched:
            mark = "  <== 当前配置" if model_id == MODEL else ""
            print(f"  - {model_id}{mark}")
    else:
        print(f"\n[注意] 没有找到包含 '{keyword}' 的模型，前 20 个模型如下:")
        for item in data[:20]:
            print(f"  - {item.get('id', '')}")
    return True

def test_chat(args):
    url = f"{BASE_URL}/chat/completions"
    print(f"\n[2/2] 最小对话测试: POST {url}")

    payload = {
        "model": args.model or MODEL,
        "messages": [{"role": "user", "content": "只回复两个字：正常"}],
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.max_tokens:
        payload["max_tokens"] = args.max_tokens

    print(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
    start = time.time()
    try:
        resp = requests.post(url, json=payload, headers=build_headers(), timeout=args.timeout)
    except requests.exceptions.RequestException as exc:
        print(f"[失败] 网络异常: {exc}")
        return False
    elapsed = time.time() - start

    print(f"状态码: {resp.status_code}  耗时: {elapsed:.1f}s")
    if resp.status_code != 200:
        print(f"响应体: {resp.text[:800]}")
        diagnose(resp.text)
        return False

    result = resp.json()
    choice = result["choices"][0]
    usage = result.get("usage", {})
    print(f"[成功] 回复: {choice['message']['content']}")
    print(f"finish_reason: {choice.get('finish_reason')}")
    print(
        f"Token: prompt={usage.get('prompt_tokens')}, "
        f"completion={usage.get('completion_tokens')}, "
        f"total={usage.get('total_tokens')}"
    )
    return True

def main():
    parser = argparse.ArgumentParser(description="外部裁判模型接口连通性检查")
    parser.add_argument("--keyword", default="gpt-6", help="模型列表过滤关键字")
    parser.add_argument("--model", default=None, help="覆盖 JUDGE_MODEL 做测试")
    parser.add_argument("--temperature", type=float, default=None, help="不传则该参数不发送")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()

    print_config()

    models_ok = True
    if not args.skip_models:
        models_ok = list_models(args.keyword)
    if not models_ok:
        print("\n[结论] 模型列表获取失败，后续对话测试大概率也会失败，仍继续尝试")

    chat_ok = True
    if not args.skip_chat:
        chat_ok = test_chat(args)

    print("\n" + "=" * 62)
    if models_ok and chat_ok:
        print("[结论] 接口可用，可以运行 judge_decision.py")
    elif chat_ok:
        print("[结论] 对话可用，但模型列表接口异常（部分中转平台不支持 /models，可忽略）")
    else:
        print("[结论] 对话测试失败，请按上方诊断建议调整 .env 后重试")

if __name__ == "__main__":
    main()