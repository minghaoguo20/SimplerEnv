import tempfile
import torch
from PIL import Image
import requests
import json
import numpy as np
import cv2
import os
import subprocess
import ast
import open3d as o3d


def create_o3d_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def obs2pcd(obs, depth_scale=1.0):
    """
    Extracts point cloud (pcd) from the observation dictionary.

    Args:
        obs (dict): The observation dictionary containing RGB, depth, and camera parameters.

    Returns:
        o3d.geometry.PointCloud: The Open3D point cloud object
    """
    import numpy as np

    rgb = obs["image"]["base_camera"]["rgb"]
    depth = obs["image"]["base_camera"]["depth"]
    seg = obs["image"]["base_camera"]["Segmentation"][..., 0]  # Extract segmentation mask
    intrinsic_matrix = obs["camera_param"]["base_camera"]["intrinsic_cv"]

    intrinsics_dict = {
        "fx": intrinsic_matrix[0, 0],
        "fy": intrinsic_matrix[1, 1],
        "ppx": intrinsic_matrix[0, 2],
        "ppy": intrinsic_matrix[1, 2],
    }

    points, colors = create_point_cloud(
        depth_image=depth, color_image=rgb, depth_scale=depth_scale, intrinsics=intrinsics_dict, seg=seg
    )

    pcd = create_o3d_point_cloud(points, colors)
    return pcd


def create_point_cloud(depth_image, color_image, depth_scale, intrinsics, use_seg=True, seg=None):
    points = []
    colors = []
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            z = depth_image[v, u] * depth_scale  # 将深度值换算成实际的距离
            if z <= 0:  # 仅处理有效的点
                points.append([0, 0, 0])
                colors.append([1.0, 0.0, 0.0])  # 归一化颜色值
                continue
            if depth_image[v, u] > 1.0:
                points.append([0, 0, 0])
                colors.append([1.0, 0.0, 0.0])  # 归一化颜色值
                continue
            if use_seg:
                if seg[v, u] > 0:
                    x = (u - intrinsics["ppx"]) * z / intrinsics["fx"]
                    y = (v - intrinsics["ppy"]) * z / intrinsics["fy"]
                    points.append([x, y, z])
                    colors.append(color_image[v, u] / 255.0)  # 归一化颜色值
                else:
                    points.append([0, 0, 0])
                    colors.append([1.0, 0.0, 0.0])  # 归一化颜色值
            else:
                x = (u - intrinsics["ppx"]) * z / intrinsics["fx"]
                y = (v - intrinsics["ppy"]) * z / intrinsics["fy"]
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)  # 归一化颜色值
    cleaned_points = []
    for idx, p in enumerate(points):
        if isinstance(p, list) and len(p) == 3:
            try:
                # 确保所有元素都是 float 或 int
                cleaned_p = [x.item() if isinstance(x, np.ndarray) else float(x) for x in p]
                cleaned_points.append(cleaned_p)
            except Exception as e:
                print(f"Skipping invalid point at index {idx}: {p}, Error: {e}")
        else:
            print(f"Invalid structure at index {idx}: {p}")

    points = np.array(cleaned_points, dtype=np.float32)
    # points = np.array(points)
    colors = np.array(colors)
    return points, colors


# def create_point_cloud(depth_image, color_image, depth_scale, intrinsics, use_seg=True, seg=None):
#     points = []
#     colors = []

#     height, width = depth_image.shape

#     for v in range(height):
#         for u in range(width):
#             z = depth_image[v, u] * depth_scale  # 深度值转换为实际距离
#             if z <= 0 or depth_image[v, u] > 1.0:  # 过滤无效深度
#                 continue

#             # 使用语义分割过滤点
#             if use_seg and seg is not None and seg[v, u] == 0:
#                 continue

#             # 计算3D点坐标
#             x = (u - intrinsics["ppx"]) * z / intrinsics["fx"]
#             y = (v - intrinsics["ppy"]) * z / intrinsics["fy"]

#             points.append([x, y, z])
#             colors.append(color_image[v, u] / 255.0)  # 颜色归一化

#     # 转换为 NumPy 数组
#     points = np.array(points, dtype=np.float32)  # (N, 3)
#     colors = np.array(colors, dtype=np.float32)  # (N, 3)

#     return points, colors


def image2depth_api(image, port=5000, temp_path="/home/xurongtao/minghao/SimplerEnv/demo/temp.jpg"):
    """
    Sends an image to the Flask server and retrieves the depth image as a NumPy array.

    Args:
        image (str, np.ndarray, list, torch.Tensor): The input image as a file path, NumPy array, list, or tensor.
        temp_path (str): The temporary path to save the image if needed.

    Returns:
        np.ndarray: The depth image in NumPy array format if the request is successful.
        None: If the request fails.
    """
    # Convert input image to file if necessary
    if isinstance(image, (np.ndarray, list)):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(temp_path, image_rgb)
        img_path = temp_path
    elif isinstance(image, str) and os.path.exists(image):
        img_path = image
    else:
        print("Invalid image input.")
        return None

    # Define the Flask API URL
    url = f"http://localhost:{port}/get_depth"

    # Create the JSON payload
    payload = {"img_path": img_path}

    try:
        # Send a POST request to the server
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

        # Check if the request was successful
        if response.status_code == 200:
            depth_image_path = response.json().get("depth_image_path")
            if depth_image_path is not None:
                # Read the depth image
                depth_data = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
                if depth_data is None:
                    print("Error: No depth data received.")
                    return None
                # Convert the depth data to a NumPy array and return
                return np.array(depth_data, dtype=np.float32)
            else:
                print("Error: No depth data received.")
                return None
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def rdt_api(
    cuda_idx="7", instruction=None, image_path=None, image_previous_path=None, depth_path=None, depth_previous_path=None
):
    # 设定 CUDA_VISIBLE_DEVICES
    env_vars = os.environ.copy()  # 复制当前环境变量
    env_vars["CUDA_VISIBLE_DEVICES"] = cuda_idx  # 设置新 GPU 设备

    # 指定 Python 解释器路径
    python_env2 = "/home/xurongtao/miniconda3/envs/rdt/bin/python"

    # 指定要运行的 Python 模块
    cmd = [
        python_env2,
        "-m",
        "scripts.afford_inference_demo_env",
        "--instruction",
        instruction,
        "--image_path",
        image_path,
        "--image_previous_path",
        image_previous_path,
        "--depth_path",
        depth_path,
        "--depth_previous_path",
        depth_previous_path,
        "--pretrained_model_name_or_path",
        "/mnt/data/xurongtao/checkpoints/rdt-finetune-1b-afford_real_qwen/checkpoint-84000/",
    ]

    # 指定工作目录
    work_dir = "/home/xurongtao/jianzhang/Afford-RDT-deploy"

    # 打印调试信息
    print(f"Running command: {' '.join(cmd)}")
    print(f"CUDA_VISIBLE_DEVICES in subprocess: {env_vars['CUDA_VISIBLE_DEVICES']}")

    # 运行命令
    result = subprocess.run(
        cmd, cwd=work_dir, env=env_vars, capture_output=True, text=True
    )

    # 打印输出，方便调试
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    return result


# def rdt_api(
#     cuda_idx="7", instruction=None, image_path=None, image_previous_path=None, depth_path=None, depth_previous_path=None
# ):
#     # 设定CUDA_VISIBLE_DEVICES
#     env_vars = {"CUDA_VISIBLE_DEVICES": cuda_idx}

#     # 指定 env2 的 Python 解释器路径
#     python_env2 = "/home/xurongtao/miniconda3/envs/rdt/bin/python"

#     # 指定要运行的 Python 模块
#     cmd = [
#         python_env2,
#         "-m",
#         "scripts.afford_inference_demo_env",
#         "--instruction",
#         instruction,  # "open the door of the cabinet",  # 指令
#         "--image_path",
#         image_path,  # "/mnt/data/Datasets/HOI4D_release/ZY20210800004/H4/C4/N42/S260/s01/T2/align_rgb/00011.jpg",  # 图像路径
#         "--image_previous_path",
#         image_previous_path,  # "/mnt/data/Datasets/HOI4D_release/ZY20210800004/H4/C4/N42/S260/s01/T2/align_rgb/00010.jpg",  # 前一帧图像路径
#         "--depth_path",
#         depth_path,  # "/mnt/data/Datasets/HOI4D_depth_video/ZY20210800004/H4/C4/N42/S260/s01/T2/align_depth/00011.png",  # 深度图路径
#         "--depth_previous_path",
#         depth_previous_path,  # "/mnt/data/Datasets/HOI4D_depth_video/ZY20210800004/H4/C4/N42/S260/s01/T2/align_depth/00010.png",  # 前一帧深度图路径
#         "--pretrained_model_name_or_path",
#         "/mnt/data/xurongtao/checkpoints/rdt-finetune-1b-afford_real_qwen/checkpoint-84000/",  # 模型权重路径
#     ]

#     # 指定工作目录
#     work_dir = "/home/xurongtao/jianzhang/Afford-RDT-deploy"

#     # 运行命令
#     result = subprocess.run(
#         cmd, cwd=work_dir, env={**env_vars, **dict(subprocess.os.environ)}, capture_output=True, text=True
#     )
#     return result


def save_images_temp(image_list, save_dir):
    """
    Save a list of images (NumPy arrays or PyTorch tensors) to a given directory temporarily.

    Args:
        image_list (list): List of images in NumPy array or PyTorch tensor format.
        save_dir (str): Directory where images should be temporarily saved.

    Returns:
        list: List of absolute file paths of the saved images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_paths = []

    for idx, img in enumerate(image_list):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()  # Convert tensor to NumPy array

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)  # Normalize if necessary

        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))  # Convert CHW to HWC if necessary

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=save_dir)
        img_path = temp_file.name
        temp_file.close()

        Image.fromarray(img).save(img_path)
        image_paths.append(os.path.abspath(img_path))

    return image_paths

def ram_api(rgb, pcd, pcd_temp_file, contact_point, post_contact_dir):
    ram_url = "http://127.0.0.1:5000/lift_affordance"
    
    o3d.io.write_point_cloud(pcd_temp_file, pcd)

    data = {
        "rgb": rgb.tolist(),
        "pcd": pcd_temp_file,
        "contact_point": contact_point,
        "post_contact_dir": post_contact_dir,
    }

    response = requests.post(ram_url, json=data)
    
    return response