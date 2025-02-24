from dataclasses import dataclass, field
import tempfile
import torch
from PIL import Image
import requests
import json
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import os
import time
import functools
import subprocess
import ast
import open3d as o3d
import sys
from io import BytesIO
import sapien.core as sapien


@dataclass
class KeyPoint:
    p2d: any = None
    p3d: any = None
    matrix: any = None

    def p3d_to_matrix(self):
        self.matrix = coordination_transform.pose_to_transformation_matrix(self.p3d)


class visualizer:
    @staticmethod
    def nparray_to_string(nparray, DIGITS):
        """
        将 NumPy 数组转换为字符串，保留指定位数的小数
        :param nparray: NumPy 数组
        :param DIGITS: 保留的小数位数
        :return: 字符串
        """
        return str(np.around(nparray.astype(np.float64), DIGITS).tolist())

    @staticmethod
    def print_dict_keys(d, indent=0):
        """
        递归遍历字典，打印所有 key 为字符串的键名，带缩进
        :param d: 需要遍历的字典
        :param indent: 当前缩进层级
        visualizer.print_dict_keys(obs)
        """
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(key, str):  # 只打印字符串类型的 key
                    if isinstance(value, np.ndarray):
                        print(" " * indent + key + ": " + value.shape.__str__())
                    else:
                        print(" " * indent + key)
                visualizer.print_dict_keys(value, indent + 4)  # 递归增加缩进层级

    @staticmethod
    def print_note_section(note, length: int = 50):
        """
        Prints a formatted note section with a border of stars.

        :param note: A string or a list of strings to display in the middle.
        :param length: The number of stars in the top and bottom border.
        """
        if isinstance(note, str):  # Convert single string to a list
            note = [note]

        max_note_length = max(len(line) for line in note)  # Find longest line in the note

        # Ensure the box is wide enough
        if length < max_note_length + 6:
            length = max_note_length + 6

        print()

        border = "*" * length
        print(border)

        for line in note:
            padding = (length - len(line) - 2) // 2
            note_line = "*" + " " * padding + line + " " * (length - len(line) - padding - 2) + "*"
            print(note_line)

        print(border)


class sim_util:
    """
    action space limit: env.unwrapped.action_space
    """

    @staticmethod
    def get_link(links, name):
        for obj in links:
            if obj.name == name:
                return obj
        return None

    @staticmethod
    def get_depth(scene: sapien.Scene, camera):
        # scene.step()  # make everything set
        scene.update_render()
        camera.take_picture()

        # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
        position = camera.get_float_texture("Position")  # [H, W, 4]

        # # OpenGL/Blender: y up and -z forward
        # points_opengl = position[..., :3][position[..., 3] < 1]
        # # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
        # # camera.get_model_matrix() must be called after scene.update_render()!
        # model_matrix = camera.get_model_matrix()
        # points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]

        depth = -position[..., 2]

        return depth

    @staticmethod
    def get_pcd_positions(camera):
        """
        get point cloud data from camera
        """
        camera.take_picture()
        # 获取相机坐标系中的点云数据
        position = camera.get_float_texture("Position")  # [H, W, 4]
        # 只保留深度值小于1的点
        points_opengl = position[..., :3][position[..., 3] < 1]
        # 获取相机的位姿并计算转换矩阵
        model_matrix = camera.get_model_matrix()  # 获取从OpenGL相机坐标系到SAPIEN世界坐标系的转换矩阵
        # 将点云转换到世界坐标系中
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        return points_world

    @staticmethod
    def extract_objects_from_env(env):
        """
        提取当前环境中所有潜在的目标物体（如 can、towel、drawer handle）。

        Args:
            env: ManiSkill2Real2Sim 的环境对象

        Returns:
            object_list: 包含所有目标物体名称的列表
        """
        target_objects = ["can", "towel", "drawer handle"]  # 目标物体关键词
        object_list = set()  # 使用 set 以自动去重

        # 1. 从 model_db 里提取可能的目标物体
        for obj_name in env.model_db.keys():
            if any(keyword in obj_name.lower() for keyword in target_objects):
                object_list.add(obj_name)

        # 2. 从 model_ids 里筛选目标物体
        for obj_name in env.model_ids:
            if any(keyword in obj_name.lower() for keyword in target_objects):
                object_list.add(obj_name)

        # 3. 解析 articulations（比如抽屉和把手）
        if hasattr(env, "get_articulations"):
            for articulation in env.get_articulations():
                if hasattr(articulation, "name") and articulation.name:
                    if any(keyword in articulation.name.lower() for keyword in target_objects):
                        object_list.add(articulation.name)

        # 4. 解析 actors（如果某些物体是单独的 actor）
        if hasattr(env, "get_actors"):
            for actor in env.get_actors():
                if hasattr(actor, "name") and actor.name:
                    if any(keyword in actor.name.lower() for keyword in target_objects):
                        object_list.add(actor.name)

        return list(object_list)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time  # 计算执行时间
        hours, rem = divmod(elapsed_time, 3600)  # 转换为小时和剩余秒数
        minutes, seconds = divmod(rem, 60)  # 转换为分钟和剩余秒数

        print(f"Function {func.__name__} took {int(hours):02}:{int(minutes):02}:{seconds:06.3f} (hh:mm:ss.sss)")
        return result

    return wrapper


def print_progress(info):
    sys.stdout.write(info)
    sys.stdout.flush()


def get_camera_name(env, camera_name=None):
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return camera_name


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

    rgb = obs["image"][camera]["rgb"]
    rgb = np.array(Image.fromarray(rgb).resize((640, 480)))
    depth = obs["image"][camera]["depth"]
    depth = np.array(Image.fromarray(depth.squeeze()).resize((640, 480)))
    seg = obs["image"][camera]["Segmentation"][..., 0]  # Extract segmentation mask
    intrinsic_matrix = obs["camera_param"][camera]["intrinsic_cv"]

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


def image2depth_api(
    image, api_url_file="demo/api_url.json", port=5001, temp_path="/home/xurongtao/minghao/SimplerEnv/demo/temp.jpg"
):
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
    # url = f"http://localhost:{port}/get_depth"
    url = load_json_data(api_url_file).get("depth")

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
        if img is None:
            image_paths.append(None)
            continue

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


def ram_api(rgb, pcd, contact_point, post_contact_dir, api_url_file="demo/api_url.json"):
    # server_url = f"http://127.0.0.1:5002/lift_affordance"
    server_url = load_json_data(api_url_file).get("ram")

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

        response = requests.post(server_url, data=data, files=files)

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
    # data = {
    #     "instruction": instruction,
    #     "image_path": image_path,
    #     "image_previous_path": image_previous_path,
    #     "depth_path": depth_path,
    #     "depth_previous_path": depth_previous_path,
    # }
    data = {
        key: value
        for key, value in {
            "instruction": instruction,
            "image_path": image_path,
            "image_previous_path": image_previous_path,
            "depth_path": depth_path,
            "depth_previous_path": depth_previous_path,
        }.items()
        if value is not None
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


def load_json_data(data_file: str):
    """Load rotation data from a JSON file."""
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return {}


class hyperparams:
    @staticmethod
    def get_hyper(task_name: str, hyper: str, data_file: str):
        """Retrieve the rotation values for a specific task from the JSON file."""
        data = load_json_data(data_file)
        return data.get(task_name, {}).get(hyper, None)

    @staticmethod
    def get_gripper_action(task_name: str, data_file: str):
        """Retrieve the gripper action for a specific task from the JSON file."""
        data = load_json_data(data_file)
        return data.get(task_name, {}).get("gripper_action", 0)

    @staticmethod
    def get_rotation(task_name: str, data_file: str):
        """Retrieve the rotation values for a specific task from the JSON file."""
        data = load_json_data(data_file)
        return data.get(task_name, {}).get("rotation", [0, 0, 0])


def depth_api(image, api_url_file="demo/api_url.json"):
    """
    发送图片到Flask服务器，获取深度图。

    :param image_path: 输入图片的路径
    :param server_url: Flask服务器的URL（默认是本地服务器）
    :return: 返回深度图（numpy 数组格式）
    """

    server_url = load_json_data(api_url_file).get("depth")

    # # 读取图像
    # image = cv2.imread(image_path)
    # if image is None:
    #     raise ValueError("无法加载输入图像，请检查路径是否正确")

    # 以二进制格式编码图片
    _, img_encoded = cv2.imencode(".jpg", image)
    files = {"image": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}

    # 发送POST请求
    response = requests.post(server_url, files=files)

    if response.status_code == 200:
        # 读取返回的深度图
        depth_image = Image.open(BytesIO(response.content))
        depth_array = np.array(depth_image)
        return depth_array
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(response.json())
        return None


class coordination_transform:
    @staticmethod
    def cal_distance(object_transformation_matrix_world, gripper_transformation_matrix_world):
        return np.linalg.norm(object_transformation_matrix_world[:3, 3] - gripper_transformation_matrix_world[:3, 3])

    @staticmethod
    def cal_angle(object_transformation_matrix_world, gripper_transformation_matrix_world):
        # Calculate the angle between gripper and object
        gripper_direction = gripper_transformation_matrix_world[:3, 2]  # Assuming the z-axis is the forward direction
        object_direction = object_transformation_matrix_world[:3, 2]  # Assuming the z-axis is the forward direction
        # Normalize the direction vectors
        gripper_direction /= np.linalg.norm(gripper_direction)
        object_direction /= np.linalg.norm(object_direction)
        # Calculate the dot product and angle
        dot_product = np.dot(gripper_direction, object_direction)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip to avoid numerical issues
        # Convert angle to degrees
        angle_degrees = np.degrees(angle)
        return angle_degrees

    @staticmethod
    def compute_action(gripper_T, target_T):
        # Extract translation vectors
        delta_xyz = target_T[:3, 3] - gripper_T[:3, 3]

        # Compute relative rotation matrix
        R_diff = target_T[:3, :3] @ np.linalg.inv(gripper_T[:3, :3])

        # Convert rotation matrix to axis-angle
        rot = Rotation.from_matrix(R_diff)
        axis_angle = rot.as_rotvec()  # Axis-angle representation

        # Concatenate translation and rotation
        action = np.hstack((delta_xyz, axis_angle))
        return action

    @staticmethod
    def quaternion_to_axis_angle(q):
        """
        Converts a quaternion to axis-angle representation.

        Args:
            q: A list or array of four elements [w, x, y, z] representing a quaternion.

        Returns:
            axis_angle: A numpy array of shape (3,), representing the rotation in axis-angle form.
        """
        w, x, y, z = q
        theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))  # 计算旋转角度，防止超出范围

        sin_half_theta = np.sqrt(1 - w**2)  # sin(theta/2) = sqrt(1 - cos^2(theta/2))

        if sin_half_theta < 1e-6:  # 处理小角度问题，防止除 0
            return np.array([0, 0, 0])  # 近似无旋转

        axis = np.array([x, y, z]) / sin_half_theta  # 计算旋转轴
        return axis * theta  # 轴角格式: 轴 * 角度

    @staticmethod
    def pose_to_transformation_matrix(input_pose):
        """
        Convert a TCP pose (position + quaternion) to a 4x4 transformation matrix in world frame.

        Args:
            tcp_pose: np.array of shape (7,), [x, y, z, qw, qx, qy, qz]

        Returns:
            T_world_tcp: np.ndarray of shape (4, 4), transformation matrix of TCP in world frame
        """

        """
        Convert a TCP pose (position + quaternion) to a 4x4 transformation matrix in world frame.

        Args:
            input_pose: Either a numpy array of shape (7,) [x, y, z, qw, qx, qy, qz],
                        or a Pose object with position (3,) and quaternion (4,).

        Returns:
            T_world_tcp: np.ndarray of shape (4, 4), transformation matrix of TCP in world frame
        """
        if isinstance(input_pose, sapien.Pose):
            position = input_pose.p
            quaternion = input_pose.q
        elif isinstance(input_pose, np.ndarray) and input_pose.shape == (7,):
            position = input_pose[:3]
            quaternion = np.array([input_pose[4], input_pose[5], input_pose[6], input_pose[3]])  # (qx, qy, qz, qw)
        else:
            raise ValueError("Invalid input: expected Pose object or np.array of shape (7,)")

        # Compute rotation matrix
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()  # (3,3) rotation matrix

        # Construct 4×4 homogeneous transformation matrix
        T_world_tcp = np.eye(4)
        T_world_tcp[:3, :3] = rotation_matrix  # Rotation part
        T_world_tcp[:3, 3] = position  # Translation part

        return T_world_tcp

        # # 提取位置
        # position = input_pose[:3]  # (x, y, z)

        # # 修正四元数顺序 (qw, qx, qy, qz) → (qx, qy, qz, qw)
        # quaternion = np.array([input_pose[4], input_pose[5], input_pose[6], input_pose[3]])  # (qx, qy, qz, qw) → (x, y, z, w)

        # # 计算旋转矩阵
        # rotation_matrix = Rotation.from_quat(quaternion).as_matrix()  # (3,3) 旋转矩阵

        # # 构造 4×4 齐次变换矩阵
        # T_world_tcp = np.eye(4)
        # T_world_tcp[:3, :3] = rotation_matrix  # 旋转部分
        # T_world_tcp[:3, 3] = position  # 平移部分

        # return T_world_tcp

    @staticmethod
    def position_to_camera(world_position_matrix, camera_extrinsic):
        """
        Convert a 3D world position to camera coordinate system.
        Args:
            world_position_matrix: np.ndarray (3,), position in world frame [x, y, z]
            camera_extrinsic: np.ndarray (4, 4), camera extrinsic matrix (world → camera)
        Returns:
            np.ndarray (3,), transformed 3D coordinates in camera space
        """
        world_pos_homogeneous = np.append(world_position_matrix, 1)  # Convert to homogeneous [x, y, z, 1]

        # Transform to camera coordinate frame
        camera_pos_homogeneous = camera_extrinsic @ world_pos_homogeneous
        return camera_pos_homogeneous[:3]  # Drop the homogeneous coordinate

    @staticmethod
    def world_to_camera(world_pose_matrix, camera_extrinsic):
        """
        Convert a 3D world pose to camera coordinate system.
        Args:
            world_pose_matrix: np.ndarray (4, 4), transformation matrix in world frame
            camera_extrinsic: np.ndarray (4, 4), camera extrinsic matrix (world → camera)
        Returns:
            np.ndarray (3,), transformed 3D coordinates in camera space
        """
        world_pos = world_pose_matrix[:3, 3]  # Extract (x, y, z) position
        world_pos_homogeneous = np.append(world_pos, 1)  # Convert to homogeneous [x, y, z, 1]

        # Transform to camera coordinate frame
        camera_pos_homogeneous = camera_extrinsic @ world_pos_homogeneous
        return camera_pos_homogeneous[:3]  # Drop the homogeneous coordinate

    @staticmethod
    def project_to_image(camera_pos, camera_intrinsic):
        """
        Project a 3D point in camera space to 2D image space.
        Args:
            camera_pos: np.ndarray (3,), position in camera space [X, Y, Z]
            camera_intrinsic: np.ndarray (3, 3), camera intrinsic matrix
        Returns:
            np.ndarray (2,), 2D image coordinates [u, v]
        """
        X, Y, Z = camera_pos
        if Z <= 0:
            return None  # Point is behind the camera

        # Apply intrinsic matrix: [u, v, 1] = K @ [X, Y, Z]
        uv_homogeneous = camera_intrinsic @ np.array([X, Y, Z])
        u, v = uv_homogeneous[:2] / uv_homogeneous[2]  # Normalize by depth (Z)

        return np.array([u, v])

    @staticmethod
    def dist_2d(point1, point2):
        """
        Calculate the 2D Euclidean distance between two points.
        Args:
            point1: np.ndarray (2,), [x1, y1]
            point2: np.ndarray (2,), [x2, y2]
        Returns:
            float, Euclidean distance between the two points
        """
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def compute_pre_pose(object_position, tcp_quaternion, d=0.1):
        """
        计算抓取位姿（grasp_pose）

        参数:
        - object_position: np.ndarray (3,) -> 物体中心的位置 (x, y, z)
        - tcp_quaternion: np.ndarray (4,) -> TCP 的方向 (w, x, y, z)
        - d: float -> 预抓取到物体的距离 (默认 0.1m)

        返回:
        - grasp_pose: Pose -> 计算出的抓取位姿 (position, orientation)
        """
        # 计算 TCP 的旋转矩阵
        tcp_quaternion_corrected = np.array(
            [tcp_quaternion[1], tcp_quaternion[2], tcp_quaternion[3], tcp_quaternion[0]]
        )
        tcp_rotation = Rotation.from_quat(tcp_quaternion_corrected).as_matrix()

        # 计算 TCP 的 Z 轴方向（抓取方向）
        tcp_forward_vector = tcp_rotation[:, 2]  # Z 轴方向

        # 计算抓取位置（沿 TCP 方向前进 d）
        grasp_position = object_position - d * tcp_forward_vector

        # 返回抓取位姿
        grasp_pose = sapien.Pose(grasp_position, tcp_quaternion)
        return grasp_pose


class test_func:
    @staticmethod
    def test_depth_api():
        depth_map = depth_api("/home/xurongtao/minghao/SimplerEnv/demo/temp.jpg")
        if depth_map is not None:
            save_path = "/home/xurongtao/minghao/SimplerEnv/demo/temp_depth_map.png"
            cv2.imwrite(save_path, depth_map)
            print(f"Depth map saved to {save_path}")


if __name__ == "__main__":
    # data_file = "/home/xurongtao/minghao/SimplerEnv/demo/simpler_data.json"
    # print(get_gripper_action("demo", data_file))  # Output: 1
    # print(get_rotation("demo", data_file))  # Output: [0.1, 0.2, 0.3]
    # print(load_json_data("demo/api_url.json").get("depth"))

    # test_func.test_depth_api()

    visualizer.print_note_section(["Test Section", "hi", "hello"])
