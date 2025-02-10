# This file is used as a API test file
# Please modify this file to make it possible to directly get the result of RDT inference
# The result should be a structured data, like numpy ndarray, torch tensor, or dict
# This python file should be able to run in this conda environment:
#   `/mnt/data/xurongtao/minghao/conda/envs/simpler_env/bin/python`

import subprocess
import ast
from debug_util import setup_debugger

def rdt_api(cuda_idx="7"):
    # 设定CUDA_VISIBLE_DEVICES
    env_vars = {"CUDA_VISIBLE_DEVICES": cuda_idx}

    # 指定 env2 的 Python 解释器路径
    python_env2 = "/home/xurongtao/miniconda3/envs/rdt/bin/python"

    # 指定要运行的 Python 模块
    cmd = [
        f"CUDA_VISIBLE_DEVICES={cuda_idx}",
        python_env2,
        "-m",
        "scripts.afford_inference_demo_env",
        "--instruction",
        "open the door of the cabinet",  # 指令
        "--image_path",
        "/mnt/data/Datasets/HOI4D_release/ZY20210800004/H4/C4/N42/S260/s01/T2/align_rgb/00011.jpg",  # 图像路径
        "--image_previous_path",
        "/mnt/data/Datasets/HOI4D_release/ZY20210800004/H4/C4/N42/S260/s01/T2/align_rgb/00010.jpg",  # 前一帧图像路径
        "--depth_path",
        "/mnt/data/Datasets/HOI4D_depth_video/ZY20210800004/H4/C4/N42/S260/s01/T2/align_depth/00011.png",  # 深度图路径
        "--depth_previous_path",
        "/mnt/data/Datasets/HOI4D_depth_video/ZY20210800004/H4/C4/N42/S260/s01/T2/align_depth/00010.png",  # 前一帧深度图路径
        "--pretrained_model_name_or_path",
        "/mnt/data/xurongtao/checkpoints/rdt-finetune-1b-afford_real_qwen/checkpoint-84000/",
    ]  # 模型权重路径

    # 指定工作目录
    work_dir = "/home/xurongtao/jianzhang/Afford-RDT-deploy"

    # 运行命令
    result = subprocess.run(
        cmd, cwd=work_dir, env={**env_vars, **dict(subprocess.os.environ)}, capture_output=True, text=True
    )
    return result


result = rdt_api()

# 打印输出结果
print("标准输出:", result.stdout)
print("=" * 50)
print("标准错误:", result.stderr)

result_dict = ast.literal_eval(result.stdout)

print(
    f"normalized_points: {result_dict['normalized_points']}"
)  # normalized_points: [[1320.0, 604.0], [1304.0, 636.0], [1120.0, 556.0], [1136.0, 560.0]]
print(f"first point: {result_dict['normalized_points'][0]}")  # first point: [1320.0, 604.0]
