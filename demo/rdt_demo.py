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
from rdt_util import (
    create_point_cloud,
    image2depth_api,
    rdt_api,
    save_images_temp,
    obs2pcd,
    ram_api,
    get_rotation,
    get_gripper_action,
)
import shutil
import tempfile


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
    retrial_times = 5

    while not (done or truncated):
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        # Resize image to 640x480
        image = np.array(Image.fromarray(image).resize((640, 480)))
        depth = image2depth_api(image=image, port=args.depth_port)

        if prev_image is None:
            prev_image = image
        if prev_depth is None:
            prev_depth = depth

        paths = save_images_temp(image_list=[image, prev_image, depth, prev_depth])

        # Use prev_image for your action model if needed
        rdt_result = rdt_api(
            instruction=instruction,
            image_path=paths[0],
            image_previous_path=paths[1],
            depth_path=paths[2],
            depth_previous_path=paths[3],
            port=args.rdt_port,
        )
        points = rdt_result["points"]

        for pathh in paths:
            os.remove(pathh)

        first_point = points[0]
        point_direction = points[1]
        # point_direction = [points[1][0] - points[0][0], points[1][1] - points[0][1]]

        pcd = obs2pcd(obs, depth_scale=1.0)  # tuple(np.array(length, 3), np.array(length, 3))
        ram_result = ram_api(
            rgb=obs["image"]["overhead_camera"]["rgb"],
            pcd=pcd,
            contact_point=first_point,
            post_contact_dir=point_direction,
            ram_url=args.ram_url,
        )  #  [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

        if ram_result is None:
            retrial_times -= 1
            if retrial_times == 0:
                break
            continue
        goal_position = ram_result["position"]  # t_obj at camera
        goal_pos_cam_hom = np.append(goal_position, 1)
        goal_pos_world_hom = obs["camera_param"]["overhead_camera"]["cam2world_gl"] @ goal_pos_cam_hom
        goal_pos_world = goal_pos_world_hom[:3]

        action = env.action_space.sample()  # replace this with your policy inference
        action[0:3] = goal_pos_world - obs["robot"]["gripper"]["position"]
        action[3:6] = get_rotation(task_name=task_name, rotation_data_file=args.rotation_data_file)
        action[6] = get_gripper_action()

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
    parser.add_argument("--ram_url", type=str, default=f"http://210.45.70.21:20606/lift_affordance", help="RAM url.")
    parser.add_argument("--rdt_port", type=int, default=5003, help="RDT Port.")
    parser.add_argument("--rotation_data_file", type=str, default="")
    args = parser.parse_args()

    main(args)
