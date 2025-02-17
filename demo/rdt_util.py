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


def obs2pcd(obs, depth_scale=1.0, camera="overhead_camera"):
    """
    Extracts point cloud (pcd) from the observation dictionary.

    Args:
        obs (dict): The observation dictionary containing RGB, depth, and camera parameters.

    Returns:
        o3d.geometry.PointCloud: The Open3D point cloud object
    """
    import numpy as np

    rgb = obs["image"]["overhead_camera"]["rgb"]
    rgb = np.array(Image.fromarray(rgb).resize((640, 480)))
    depth = obs["image"]["overhead_camera"]["depth"]
    depth = np.array(Image.fromarray(depth.squeeze()).resize((640, 480)))
    seg = obs["image"]["overhead_camera"]["Segmentation"][..., 0]  # Extract segmentation mask
    intrinsic_matrix = obs["camera_param"]["overhead_camera"]["intrinsic_cv"]

    intrinsics_dict = {
        "fx": intrinsic_matrix[0, 0],
        "fy": intrinsic_matrix[1, 1],
        "ppx": intrinsic_matrix[0, 2],
        "ppy": intrinsic_matrix[1, 2],
    }

    points, colors = create_point_cloud(
        depth_image=depth, color_image=rgb, depth_scale=depth_scale, intrinsics=intrinsics_dict, use_seg=False, seg=seg
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


def image2depth_api(image, port=5001, temp_path="/home/xurongtao/minghao/SimplerEnv/demo/temp.jpg"):
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


def rdt_cmd(
    cuda_idx="7", instruction=None, image_path=None, image_previous_path=None, depth_path=None, depth_previous_path=None
):
    """
    client demo:
    ```
    rdt_result = rdt_cmd(
        cuda_idx=args.rdt_cuda,
        instruction=instruction,
        image_path=paths[0],
        image_previous_path=paths[1],
        depth_path=paths[2],
        depth_previous_path=paths[3],
    )
    rdt_result = ast.literal_eval(rdt_result.stdout)
    ````
    """
    # 设定 CUDA_VISIBLE_DEVICES
    env_vars = os.environ.copy()  # 复制当前环境变量
    env_vars["CUDA_VISIBLE_DEVICES"] = cuda_idx  # 设置新 GPU 设备

    # 指定 Python 解释器路径
    python_env2 = "/home/xurongtao/miniconda3/envs/rdt/bin/python"

    # 指定要运行的 Python 模块
    cmd = [
        python_env2,
        "-m",
        "scripts.afford_inference_demo_env_minghao",
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
    result = subprocess.run(cmd, cwd=work_dir, env=env_vars, capture_output=True, text=True)

    # 打印输出，方便调试
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    return result


def save_images_temp(image_list):
    """
    Save a list of images (NumPy arrays or PyTorch tensors) to a given directory temporarily.

    Args:
        image_list (list): List of images in NumPy array or PyTorch tensor format.
        save_dir (str): Directory where images should be temporarily saved.

    Returns:
        list: List of absolute file paths of the saved images.
    """
    image_paths = []

    for idx, img in enumerate(image_list):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()  # Convert tensor to NumPy array

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)  # Normalize if necessary

        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))  # Convert CHW to HWC if necessary

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img_path = temp_file.name
        temp_file.close()

        Image.fromarray(img).save(img_path)
        image_paths.append(os.path.abspath(img_path))

    return image_paths


def ram_api(rgb, pcd, contact_point, post_contact_dir, ram_url=f"http://210.45.70.21:20606/lift_affordance"):
    # ram_url = f"http://127.0.0.1:5002/lift_affordance"

    # 使用临时文件存储 PCD
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as temp_pcd:
        pcd_temp_path = temp_pcd.name
        o3d.io.write_point_cloud(pcd_temp_path, pcd)

    # 发送数据
    with open(pcd_temp_path, "rb") as pcd_file:
        files = {"pcd": pcd_file}
        data = {
            "rgb": json.dumps(rgb.tolist()),  # 转换为 JSON 字符串
            "contact_point": json.dumps(contact_point),  # 转换为 JSON
            "post_contact_dir": json.dumps(post_contact_dir),  # 转换为 JSON
        }

        response = requests.post(ram_url, data=data, files=files)

    # 删除临时 PCD 文件
    os.remove(pcd_temp_path)

    if response.status_code == 200:
        result = response.json()
        if len(result) == 1 and "error" in result:
            print(f"Error: {result['error']}")
            return None
        return result
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def rdt_api(
    instruction=None, image_path=None, image_previous_path=None, depth_path=None, depth_previous_path=None, port=5003
):
    url = f"http://localhost:{port}/inference"
    data = {
        "instruction": instruction,
        "image_path": image_path,
        "image_previous_path": image_previous_path,
        "depth_path": depth_path,
        "depth_previous_path": depth_previous_path,
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        if len(result) == 1 and "error" in result:
            print(f"Error: {result['error']}")
            return None
        return result
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def load_rotation_data(data_file: str):
    """Load rotation data from a JSON file."""
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return {}


def get_gripper_action(task_name: str, data_file: str):
    """Retrieve the gripper action for a specific task from the JSON file."""
    data = load_rotation_data(data_file)
    return data.get(task_name, {}).get("gripper_action", 0)


def get_rotation(task_name: str, data_file: str):
    """Retrieve the rotation values for a specific task from the JSON file."""
    data = load_rotation_data(data_file)
    return data.get(task_name, {}).get("rotation", [0, 0, 0])


if __name__ == "__main__":
    data_file = "/home/xurongtao/minghao/SimplerEnv/demo/simpler_data.json"
    print(get_gripper_action("demo", data_file))  # Output: 1
    print(get_rotation("demo", data_file))  # Output: [0.1, 0.2, 0.3]
