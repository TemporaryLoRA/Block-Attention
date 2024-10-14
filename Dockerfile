FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ " \
    LANG="zh_CN.UTF-8" \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_IB_HCA="rdma_0:1,rdma_1:1,rdma_2:1,rdma_3:1" \
    NCCL_DEBUG="INFO" \
    NCCL_DEBUG_SUBSYS="init,net,graph,env,tuning" \
    NCCL_IB_GID_INDEX=3 \
    NCCL_IB_QPS_PER_CONNECTION=10 \
    NCCL_IB_TIMEOUT=23 \
    NCCL_IB_RETRY_CNT=7 \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    CUDA_HOME="/usr/local/cuda-11.8" \
    CUTLASS_PATH="/root/.local/cutlass-3.5.0"

RUN apt update && \
    apt upgrade -y && \
    apt install apt-utils -y && \
    dpkg --configure -a && \
    apt install libaio-dev -y && \
    apt install net-tools -y && \
    apt install zip -y && \
    apt install unzip -y && \
    apt install pdsh -y && \
    apt install git git-lfs -y && \
    apt install vim -y && \
    apt install libreadline8 -y && \
    apt install language-package-zh-hans -y && \
    apt install openssh-server -y && \
    apt install cmake -y && \
    apt install tmux -y && \
    apt install mpich -y && \
    apt install python3 -y && \
    apt install python3-setuptools -y && \
    apt install python3-pip -y  && \
    apt install apt-transport-https ca-certificates gnupg curl -y && \
    apt update && \
    apt upgrade -y && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN pip install pip --upgrade && \
    pip install numpy pandas scikit-learn --upgrade && \
    pip install fastapi flask flask-socket-io flask_cors unicorn gunicorn werkzeug pymongo redis --upgrade &&  \
    pip install coloredlogs colorama termcolor --upgrade && \
    pip install ipdb --upgrade && \
    pip install joblib --upgrade && \
    pip install pyarrow --upgrade && \
    pip install psutil --upgrade && \
    pip install fire --upgrade && \
    pip install pydantic --upgrade && \
    pip install pyecharts --upgrade && \
    pip install overrides --upgrade && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install transformers peft evaluate accelerate trl --upgrade && \
    pip install bitsandbytes --upgrade && \
    pip install sentencepiece --upgrade && \
    pip install safetensors --upgrade && \
    pip install torchmetrics --upgrade && \
    pip install einops --upgrade && \
    pip install ray --upgrade && \
    pip install openpyxl --upgrade && \
    pip install autoawq --upgrade && \
    pip install mpi4py --upgrade && \
    pip install pip --upgrade


RUN pip install cupy-cuda11x && \
    pip install deepspeed --upgrade && \
    pip install deepspeed-kernels && \
    pip install aioprometheus && \
    MAX_JOBS=64 pip install flash-attn --no-build-isolation && \
    pip install xformers --index-url https://download.pytorch.org/whl/cu118 --no-deps && \
    pip install vllm --upgrade --no-deps


CMD ["/bin/bash"]
WORKDIR "/root"


# cutlass







