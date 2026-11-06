# ============================================================
#  多架构基础镜像选择
#  - GPU 场景：nvidia/cuda:12.1.0-runtime-ubuntu22.04（约 1.2GB）
#  - CPU 场景：python:3.11-slim（约 50MB，推荐用于减小镜像）
# ============================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# FROM python:3.11-slim  # 无 GPU 时取消注释此行，镜像体积可减小约 90%

# ========== 环境变量 ==========
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TORCH_CUDA_ARCH_LIST=Auto

# ========== 系统依赖（最小化安装，合并单层） ==========
# 仅保留 Python 3.11 运行时
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# ========== pip（Python 3.11 专用） ==========
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ========== 工作目录 ==========
WORKDIR /app

# ========== 安装 Python 依赖（先复制 requirements.txt 以利用 Docker 缓存层） ==========
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ========== 复制项目代码（仅复制必需的 .py 文件） ==========
COPY *.py ./

# ========== 默认命令 ==========
CMD ["bash"]





#构建镜像命令：docker build -t location-value-inference:vlee1.0 .
# 运行方式：
  #有 GPU：docker run --gpus all -it --rm your-image
  #无 GPU：docker run -it --rm your-image（不加 --gpus 参数）
    # 即使使用统一版本，在无 GPU 机器上 PyTorch 会自动 fallback 到 CPU 模式，但代码中需要确保有设备检测逻辑：
    #import torch
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

