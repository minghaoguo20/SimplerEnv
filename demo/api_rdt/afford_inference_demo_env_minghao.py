import time
import yaml
import argparse
import os
import copy, json


import torch

import numpy as np
from PIL import Image as PImage
from PIL import Image, ImageDraw

from scripts.afford_model import create_model
from data.point_vla_dataset import AffordVLADataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import cv2

model_path = (
    "/mnt/data/xurongtao/checkpoints/rdt-finetune-1b-afford_real_qwen/checkpoint-84000/"
)
pretrained_model_name_or_path = "/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer/checkpoints/rdt-finetune-1b-afford6_1/checkpoint-80000"


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Random seed",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )

    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )

    parser.add_argument(
        "--puppet_arm_left_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_left_cmd_topic",
        default="/master/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_cmd_topic",
        action="store",
        type=str,
        help="puppet_arm_right_cmd_topic",
        default="/master/joint_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_topic",
        action="store",
        type=str,
        help="puppet_arm_left_topic",
        default="/puppet/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_topic",
        action="store",
        type=str,
        help="puppet_arm_right_topic",
        default="/puppet/joint_right",
        required=False,
    )

    parser.add_argument(
        "--robot_base_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/odom_raw",
        required=False,
    )
    parser.add_argument(
        "--robot_base_cmd_topic",
        action="store",
        type=str,
        help="robot_base_topic",
        default="/cmd_vel",
        required=False,
    )
    parser.add_argument(
        "--use_robot_base",
        action="store_true",
        help="Whether to use the robot base to move around",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--ctrl_freq",
        action="store",
        type=int,
        help="The control frequency of the robot",
        default=25,
        required=False,
    )

    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=64,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        help="The maximum change allowed for each joint per timestep",
        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
        required=False,
    )

    parser.add_argument(
        "--use_actions_interpolation",
        action="store_true",
        help="Whether to interpolate the actions if the difference is too large",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--disable_puppet_arm",
        action="store_true",
        help="Whether to disable the puppet arm. This is useful for safely debugging",
        default=False,
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/base.yaml",
        help="Path to the config file",
    )

    parser.add_argument("--instruction", type=str, required=True, help="Instruction")

    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image"
    )
    parser.add_argument(
        "--image_previous_path",
        type=str,
        required=True,
        help="Path to the previous image",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        required=False,
        default=None,
        help="Path to the depth image",
    )
    parser.add_argument(
        "--depth_previous_path",
        type=str,
        required=False,
        default=None,
        help="Path to the previous depth image",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=False,
        default="/home/panwen.hu/workspace/jian.zhang/EAI/AffordDiffusionTransformer/checkpoints/rdt-finetune-1b-afford6_1/checkpoint-80000",
        help="Name or path to the pretrained model",
    )

    # parser.add_argument('--lang_embeddings_path', type=str, required=True,
    #                     help='Path to the pre-encoded language instruction embeddings')

    args = parser.parse_args()
    return args


def draw_arrows_on_image(image_array, points):
    # 检查输入点的格式
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("Points tensor must have shape (N, 2).")

    # 将 PyTorch 张量转换为 NumPy 数组
    points_np = points.to(dtype=torch.float32).cpu().numpy().astype(int)  # 确保整数类型
    # points_np = points

    # 确保图片是 NumPy 数组
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be a numpy.ndarray.")

    # 遍历点，绘制箭头
    for i in range(len(points_np) - 1):
        pt1 = tuple(points_np[i])  # 起始点 (x1, y1)
        pt2 = tuple(points_np[i + 1])  # 终点 (x2, y2)
        # 绘制箭头
        cv2.arrowedLine(
            image_array, pt1, pt2, color=(0, 0, 255), thickness=2, tipLength=0.3
        )  # 绿色箭头

    return image_array


def draw_arrows_on_image_cv2(image_array, points, save_path="output_image.jpg"):
    """
    使用 OpenCV 在图片上绘制箭头并保存。

    Args:
        image_array (numpy.ndarray): 图片数组，形状为 (H, W, 3)。
        points (torch.Tensor): 包含点的张量，形状为 (N, 2)，表示 N 个点的 x, y 坐标。
        save_path (str): 保存图片的路径。
    """
    # 检查输入点的格式
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("Points tensor must have shape (N, 2).")

    # 将 PyTorch 张量转换为 NumPy 数组
    points_np = points.to(dtype=torch.float32).cpu().numpy().astype(int)  # 确保整数类型

    # 确保图片是 NumPy 数组
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be a numpy.ndarray.")

    # 遍历点，绘制箭头
    for i in range(len(points_np) - 1):
        pt1 = tuple(points_np[i])  # 起始点 (x1, y1)
        pt2 = tuple(points_np[i + 1])  # 终点 (x2, y2)
        # 绘制箭头
        cv2.arrowedLine(
            image_array, pt1, pt2, color=(0, 255, 0), thickness=2, tipLength=0.3
        )  # 绿色箭头

    # 保存图片
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, image_array)
    # print(f"Image with arrows saved to {save_path}")


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    font_scale: float = 1.0,
    color: tuple = (0, 0, 0),
    thickness: int = 5,
) -> np.ndarray:
    """
    Draw text on the top-right corner of a given image (NumPy array).

    Parameters:
        image (np.ndarray): The input image (H, W, C) as a NumPy array.
        text (str): The text to draw.
        font_scale (float): Font size for the text.
        color (tuple): Text color in BGR (default is white).
        thickness (int): Thickness of the text.

    Returns:
        np.ndarray: Image with the text drawn on it.
    """
    # Make a copy of the image to avoid modifying the original
    image_with_text = image.copy()

    # Define the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the size of the text box
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the position for the text (right-top corner)
    x = image_with_text.shape[1] - text_size[0] - 10  # 10px padding from the right edge
    y = text_size[1] + 10  # 10px padding from the top edge

    # Draw the text on the image
    cv2.putText(image_with_text, text, (x, y), font, font_scale, color, thickness)

    return image_with_text


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # result.paste(pil_img, (0, (width - height) // 2))
        result.paste(pil_img, (0, 0))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        # result.paste(pil_img, ((height - width) // 2, 0))
        result.paste(pil_img, (0, 0))
        return result


# Initialize the model
def make_policy(model_path, pretrained_model_name_or_path):
    with open(model_path, "r") as fp:
        config = yaml.safe_load(fp)

    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=25,
    )

    return model


# RDT inference
def inference_fn(policy, args):

    # pretrained_models = ['/mnt/data/xurongtao/checkpoints/rdt-finetune-1b-afford_real_qwen/checkpoint-84000/',]
    # args.pretrained_model_name_or_path = pretrained_models[0] # 选择预训练模型

    # 'open the door of the cabinet'
    instruction = args.instruction  # 指令

    # '/mnt/data/Datasets/HOI4D_release/ZY20210800004/H4/C4/N42/S260/s01/T2/align_rgb/00011.jpg'
    image_path = args.image_path  # 输入图像
    assert os.path.exists(image_path)

    # image_previous_path = None
    # '/mnt/data/Datasets/HOI4D_release/ZY20210800004/H4/C4/N42/S260/s01/T2/align_rgb/00010.jpg'
    image_previous_path = args.image_previous_path  # 前一帧图片
    assert os.path.exists(image_previous_path)

    # depth_path = None
    # '/mnt/data/Datasets/HOI4D_depth_video/ZY20210800004/H4/C4/N42/S260/s01/T2/align_depth/00011.png'
    depth_path = args.depth_path  # 深度图
    # '/mnt/data/Datasets/HOI4D_depth_video/ZY20210800004/H4/C4/N42/S260/s01/T2/align_depth/00010.png'
    depth_previous_path = args.depth_previous_path

    # 可视化图片保存路径生成
    dir_name, file_name = os.path.split(image_path)
    file_base, file_ext = os.path.splitext(file_name)
    # 生成新的文件名
    new_file_name = f"{file_base}_out{file_ext}"
    # 组合新的路径
    new_image_path = os.path.join(dir_name, new_file_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_previous_path:
        image_previous = cv2.imread(image_previous_path)
        image_previous = cv2.cvtColor(image_previous, cv2.COLOR_BGR2RGB)
    else:
        image_previous = None

    if depth_path:
        depth = cv2.imread(depth_path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    else:
        depth = None
    if depth_previous_path:
        depth_previous = cv2.imread(depth_previous_path)
        depth_previous = cv2.cvtColor(depth_previous, cv2.COLOR_BGR2RGB)
    else:
        depth_previous = None

    # 模型推理部分
    # Load rdt model
    # policy = make_policy(args)

    lang_embeddings = policy.encode_instruction(instruction)

    time1 = time.time()
    # fetch images in sequence [front, right, left]
    image_arrs = [
        image_previous,
        depth_previous,
        None,
        image,
        depth,
        None,
    ]

    images2 = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]

    # adapt to RDT format
    # proprio = proprio.unsqueeze(0)
    proprio = torch.zeros(1, 14)
    states = np.zeros((1, 128))
    state_indicator = np.zeros(128)
    state_indicator[103:105] = 1

    # actions shaped as [1, 64, 14] in format [left, right]
    actions = policy.step(
        proprio=proprio,
        images=images2,
        text_embeds=lang_embeddings,
        states=states,
        state_indicator=state_indicator,
    )

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    normalized_points = actions[0, 0:4, 103:105]
    normalized_points = normalized_points.view(-1, 2)
    points = actions[0, 0:4, 103:105]

    h, w, _ = image.shape

    points[:, 0] *= w
    points[:, 1] *= h

    points = points.to(torch.float)

    points = points.view(-1, 2)
    points = points.to(torch.int)

    image = draw_text_on_image(image, instruction)
    draw_arrows_on_image_cv2(image, points, save_path=new_image_path)
    result = {
        "image_shape": (h, w, 3),
        "normalized_points": normalized_points.tolist(),
        "points": points.tolist(),
    }
    print(result)
    return normalized_points, points


if __name__ == "__main__":
    policy = make_policy(
        model_path=model_path,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )
    args = get_arguments()
    normalized_points, points = inference_fn(policy, args)
