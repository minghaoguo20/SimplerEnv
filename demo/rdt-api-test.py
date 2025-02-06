# This file is used as a API test file
# Please modify this file to make it possible to directly get the result of RDT inference
# The result should be a structured data, like numpy ndarray, torch tensor, or dict
# This python file should be able to run in this conda environment:
#   `/mnt/data/xurongtao/minghao/conda/envs/simpler_env/bin/python`

import subprocess

# 设定CUDA_VISIBLE_DEVICES
env_vars = {"CUDA_VISIBLE_DEVICES": "0"}

# 指定 env2 的 Python 解释器路径
python_env2 = "/home/xurongtao/miniconda3/envs/rdt/bin/python"

# 指定要运行的 Python 模块
cmd = [python_env2, "-m", "scripts.afford_inference_demo_env"]

# 指定工作目录
work_dir = "/home/xurongtao/jianzhang/Afford-RDT-deploy"

# 运行命令
result = subprocess.run(cmd, cwd=work_dir, env={**env_vars, **dict(subprocess.os.environ)}, capture_output=True, text=True)

# 打印输出结果
print("标准输出:", result.stdout)
print("="*50)
print("标准错误:", result.stderr)
