import requests
import os
from dotenv import load_dotenv
import time

def server_llm_invoke(prompt: str = "Hello, respond with one word.") -> str:
    """
    通过 HTTP request 直接测试 Agnes-2.0-flash 模型
    """
    api_key = os.environ.get("OPENAI_API_KEY", "sk-REPLACED")
    # url = "https://apihub.agnes-ai.com/v1/chat/completions"
    
    timeout = float(os.environ.get("LLM_TIMEOUT", "600"))
    retries = int(os.environ.get("LLM_RETRIES", "3"))
    max_tokens = os.environ.get("LLM_MAX_TOKENS")

    
    url = "http://172.16.41.254:30245/v1/chat/completions"  # 本地模型接口地址

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        # "model": "agnes-2.0-flash",
        "model": "/root/kube-user/Qwen3.8-27B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # "max_tokens": 256,
    }
    
    payload["max_tokens"] = 16384
    
    # if max_tokens:
    #     payload["max_tokens"] = int(max_tokens)
    # else:
    #     payload["max_tokens"] = 2048

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()

        # 提取核心返回内容
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        print(f" 模型回复: {content}")
        print(f" Token用量: prompt={usage.get('prompt_tokens')}, "
              f"completion={usage.get('completion_tokens')}, "
              f"total={usage.get('total_tokens')}")

        time.sleep(1)

        return content

    except requests.exceptions.HTTPError as e:
        print(f" API请求失败 [{e.response.status_code}]: {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        print(f" 网络请求异常: {e}")
        raise

load_dotenv()
if __name__ == "__main__":
    # # 从 .env 文件加载环境变量（OPENAI_API_KEY 放在 .env 中）
    # load_dotenv()
    # if not os.environ.get("OPENAI_API_KEY"):
    #     raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")


    server_llm_invoke("用一句话介绍你自己")

