from debug_util import setup_debugger

if __name__ == "__main__":
    setup_debugger(ip_addr="127.0.0.1", port=9501, debug=False)

import argparse
import site
import os
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy
import sapien.core as sapien
from transformers import pipeline

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import ast
import open3d as o3d
from rdt_util import create_point_cloud, image2depth_api, rdt_api, save_images_temp, obs2pcd, ram_api


def main(args):
    site.main()

    task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

    if "env" in locals():
        print("Closing existing env")
        env.close()
        del env
    env = simpler_env.make(task_name)
    sapien.render_config.rt_use_denoiser = False
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print("Reset info", reset_info)
    print("Instruction", instruction)

    frames = []
    done, truncated = False, False
    prev_image = None  # Initialize previous image variable
    prev_depth = None  # Initialize previous image variable

    while not (done or truncated):
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        depth = image2depth_api(image=image, port=args.depth_port)

        if prev_image is None:
            prev_image = image
        if prev_depth is None:
            prev_depth = depth

        paths = save_images_temp(
            image_list=[image, prev_image, depth, prev_depth],
            save_dir="/home/xurongtao/minghao/SimplerEnv/demo/temp_save",
        )

        # Use prev_image for your action model if needed
        rdt_result = rdt_api(
            cuda_idx=args.rdt_cuda,
            instruction=instruction,
            image_path=paths[0],
            image_previous_path=paths[1],
            depth_path=paths[2],
            depth_previous_path=paths[3],
        )
        points = ast.literal_eval(rdt_result.stdout)["normalized_points"]
        first_point = points[0]
        point_direction = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
        # print(first_point)
        # print(point_direction)

        pcd = obs2pcd(obs, depth_scale=1.0)  # tuple(np.array(length, 3), np.array(length, 3))
        ram_result = ram_api(
            rgb=obs["image"]["base_camera"]["rgb"], pcd=pcd, contact_point=first_point, post_contact_dir=point_direction
        ) #  [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        # {'grasp_array': [0.04069748520851135, 0.07440207153558731, 0.019999999552965164, 0.029999999329447746, 0.93841952085495, -0.19488315284252167, -0.2852882146835327, 0.27932843565940857, -0.058008644729852676, 0.9584417939186096, -0.20333333313465118, -0.9791095852851868, -4.2798237842589515e-08, 0.7740446925163269, 0.3493576943874359, -0.7173059582710266, -1.0], 'post_grasp_dir': [0.06475112622672671, 0.8098853157989903, 0.14931714953183362]}
        action = env.action_space.sample()  # replace this with your policy inference

        obs, reward, done, truncated, info = env.step(action)

        prev_image = image  # Update prev_image to current image
        prev_depth = depth  # Update prev_depth to current depth
        frames.append(image)

    episode_stats = info.get("episode_stats", {})
    print("Episode stats", episode_stats)
    # Save the video instead of showing it
    video_save_path = "output/demo-rdt.mp4"
    mediapy.write_video(video_save_path, frames, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RDT demo.")
    parser.add_argument("--depth_port", type=int, default=5001, help="Depth Anything v2 Port.")
    # parser.add_argument("--depth_cuda", type=int, default=6, help="Depth Anything v2 Port.")
    parser.add_argument("--ram_port", type=int, default=5002, help="RAM Port.")
    # parser.add_argument("--ram_cuda", type=int, default=6, help="RAM Port.")
    parser.add_argument("--rdt_cuda", type=str, default="7", help="RAM Port.")
    parser.add_argument(
        "--pcd_temp_file",
        type=str,
        default="/home/xurongtao/minghao/SimplerEnv/demo/temp_save/point_cloud_temp.pcd",
        help="",
    )
    args = parser.parse_args()

    main(args)
