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
from PIL import Image, ImageDraw, ImageFont
import requests
import ast
import open3d as o3d
from rdt_util import (
    create_point_cloud,
    depth_api,
    rdt_api,
    save_images_temp,
    obs2pcd,
    ram_api,
    get_rotation,
    get_gripper_action,
    print_progress,
    get_camera_name,
)
import shutil
import tempfile
from datetime import datetime
import sys
import warnings

from debug_util import setup_debugger

if __name__ == "__main__":
    setup_debugger(ip_addr="127.0.0.1", port=9501, debug=False)

warnings.filterwarnings("ignore")

date_now = datetime.now()
run_date = date_now.strftime("%Y%m%d_%H%M%S")
depth_scale = 1


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_idx
    site.main()

    task_info = run_date + "_" + args.note if len(args.note) > 0 else run_date
    task_name = "google_robot_pick_object"  # @param ["google_robot_pick_coke_can", "google_robot_pick_object", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_place_in_closed_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

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
    frames_save_dir = os.path.join(args.output_dir, f"{task_name}_{task_info}")
    os.makedirs(frames_save_dir, exist_ok=True)
    print(f"Saving frames to: {frames_save_dir}")
    print(f"Time: {date_now.strftime('%H%M%S')}")

    done, truncated = False, False
    prev_image = None  # Initialize previous image variable
    prev_depth = None  # Initialize previous image variable
    retrial_times = 5

    current_step = 0
    image_info = ""
    pose_action_pose = {}
    first_point = [0, 0]
    point_direction = [0, 0]
    last_pose = obs["extra"]["tcp_pose"][:3]
    while not (done or truncated) and retrial_times > 0:
        # action[:3]: delta xyz;
        # action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill2_obs_dict(env, obs)

        # ############################## visulization ##############################
        # Save the current image to frames_save_dir
        # image_info += f'current pose: {obs["extra"]["tcp_pose"][:3]}'
        if current_step > 0:
            frames_are_different = not np.all(obs["extra"]["tcp_pose"][:3] == last_pose)
            if frames_are_different:
                image_save_path = os.path.join(frames_save_dir, f"frame_{current_step:04d}.png")
                # Write image_info on the image
                image_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(image_pil)
                draw.text((10, 10), image_info, fill="white")

                # Draw an arrow from first_point to point_direction
                start_point = (int(first_point[0]), int(first_point[1]))
                end_point = (int(point_direction[0]), int(point_direction[1]))
                draw.line([start_point, end_point], fill="red", width=3)
                draw.polygon(
                    [end_point, (end_point[0] - 5, end_point[1] - 5), (end_point[0] + 5, end_point[1] - 5)], fill="red"
                )
                image_pil.save(image_save_path)
            pose_action_pose[f"{current_step:03d}"] = {
                "frames_are_different": frames_are_different,
                "info": f'{last_pose} -> {action.tolist()} -> {obs["extra"]["tcp_pose"][:3]}',
            }

        image_info = task_info + "\n"
        # ############################################################################

        # ############################## image preprocess ##############################
        # Resize image to 640x480
        image = np.array(Image.fromarray(image).resize((640, 480)))
        depth = depth_api(image=image, api_url_file=args.api_url_file)
        # ##############################################################################

        # ############################## RDT ##############################
        if prev_image is None:
            prev_image = image
        if prev_depth is None:
            prev_depth = depth
        paths = save_images_temp(image_list=[image, prev_image, depth, prev_depth])
        
        # Use prev_image for your action model if needed
        if current_step == 0:
            rdt_result = rdt_api(
                instruction=instruction,
                image_path=paths[0],
                image_previous_path=paths[1],
                depth_path=paths[2],
                depth_previous_path=paths[3],
                port=args.rdt_port,
            )
        try:
            points = rdt_result["points"]
        except Exception as e:
            print("Error in RDT API:", e)
            retrial_times -= 1
            continue
        for pathh in paths:
            os.remove(pathh)
        # #################################################################

        first_point = points[0]
        point_direction = points[1]
        # point_direction = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
        image_info += f"RDT: {points}\n"
        image_info += f"\tfirst_point: {first_point}\n"
        image_info += f"\tpoint_direction: {point_direction}\n"

        # ############################## RAM ##############################
        pcd = obs2pcd(obs, depth_scale=depth_scale, camera=get_camera_name(env))  # depth_scale = 2
        ram_result = ram_api(
            rgb=obs["image"][get_camera_name(env)]["rgb"],
            pcd=pcd,
            contact_point=first_point,
            post_contact_dir=point_direction,
            api_url_file=args.api_url_file
        )  # [x, y, z]  # it was supposed to be: [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        if ram_result is None:
            retrial_times -= 1
            continue
        # #################################################################

        """
        for depth_scale in [30,40,50,100]:
            pcd = obs2pcd(obs, depth_scale=depth_scale, camera=get_camera_name(env))
            ram_result = ram_api(
                rgb=obs["image"][get_camera_name(env)]["rgb"],
                pcd=pcd,
                contact_point=first_point,
                post_contact_dir=point_direction,
                api_url_file=args.api_url_file
            )
            print(f"depth_scale={depth_scale} \t| xyz={ram_result['position']}")

            #       depth_scale=0.01    | xyz=[0, 0, 0]
            #       depth_scale=0.1     | xyz=[0, 0, 0]
            #       depth_scale=0.5     | xyz=[0, 0, 0]
            #       depth_scale=1       | xyz=[0, 0, 0]
            #       depth_scale=5       | xyz=[1, 0, 2]
            #       depth_scale=10      | xyz=[2, 0, 4]
            #       depth_scale=20      | xyz=[4, 1, 9]
            #       depth_scale=30      | xyz=[6, 2, 13]
            #       depth_scale=40      | xyz=[8, 3, 18]
            #       depth_scale=50      | xyz=[10, 4, 23]
        """

        # ############################## coordination convertion ##############################
        goal_position = ram_result["position"]  # t_obj at camera
        goal_pos_cam_hom = np.append(goal_position, 1)
        goal_pos_world_hom = obs["camera_param"][get_camera_name(env)]["cam2world_gl"] @ goal_pos_cam_hom
        goal_pos_world = goal_pos_world_hom[:3]
        """
        image_info += f'RAM\n'
        image_info += f'    RAM position: {ram_result["position"]}\n'
        # image_info += f'\tcam 2 world: {obs["camera_param"][get_camera_name(env)]["cam2world_gl"]}\n'
        image_info += f"    -> base coordinate: {goal_pos_world}\n"
        """
        # random_action = env.action_space.sample()  # replace this with your policy inference
        action = np.zeros(7)
        action[0:3] = goal_pos_world - obs["extra"]["tcp_pose"][:3]
        action[3:6] = get_rotation(task_name=task_name, data_file=args.simpler_data_file)
        action[6] = get_gripper_action(task_name=task_name, data_file=args.simpler_data_file)
        # ######################################################################################

        last_pose = obs["extra"]["tcp_pose"][:3]

        # action = np.zeros(7)
        # step_0 = 0.1
        # action[0] = np.exp(-step_0 * current_step)
        # image_info += f"action: {action.tolist()}\n"
        # image_info += f"current pose: {last_pose}\n"

        """
        # image_info += f"[ref] random_action: {action}\n"
        image_info += f"[Action]: \t{action.tolist()}\n"
        image_info += f'[base coordinate]: \t{obs["extra"]["tcp_pose"][:3]} -> {goal_pos_world}\n'
        # image_info += f"rotation: {action[3:6]}\n"
        # image_info += f"gripper: {action[6]}\n"
        """
        obs, reward, done, truncated, info = env.step(action)

        prev_image = image  # Update prev_image to current image
        prev_depth = depth  # Update prev_depth to current depth
        frames.append(image)

        # print_progress(
        #     f'\rFinished step: {current_step} | action: {action} | {obs["extra"]["tcp_pose"][:3]} -> {goal_pos_world}\t'
        # )
        print_progress(
            f'\rFinished step: {current_step} | action: {action.tolist()} | {obs["extra"]["tcp_pose"][:3]}\t'
        )
        current_step += 1

    episode_stats = info.get("episode_stats", {})
    print("Episode stats", episode_stats)
    # Save the video instead of showing it
    video_save_path = os.path.join(frames_save_dir, f"{task_name}_{run_date}.mp4")
    mediapy.write_video(video_save_path, frames, fps=10)
    print(f"Video saved to {video_save_path}")
    env.close()
    # Save pose_action_pose to a text file
    pose_action_save_path = os.path.join(frames_save_dir, f"{task_name}_{run_date}_pose_action_pose.txt")
    with open(pose_action_save_path, "w") as f:
        for step in pose_action_pose:
            item = pose_action_pose[step]
            if item["frames_are_different"]:
                f.write(f'{step} \t{item["info"]} + \n')
        f.write("\n" + "*" * 50 + "\n\n")
        for step in pose_action_pose:
            item = pose_action_pose[step]
            changed = "[Changed]" if item["frames_are_different"] else "[No movement]"
            f.write(f'{step} \t{changed} \t{item["info"]} + \n')
        f.write("\n" + "*" * 50 + "\n\n")
    print(f"Pose action data saved to {pose_action_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RDT demo.")
    parser.add_argument("--depth_port", type=int, default=5001, help="Depth Anything v2 Port.")
    parser.add_argument("--ram_url", type=str, default=f"http://210.45.70.21:20606/lift_affordance", help="RAM url.")
    parser.add_argument("--rdt_port", type=int, default=5003, help="RDT Port.")
    parser.add_argument(
        "--simpler_data_file", type=str, default="/home/xurongtao/minghao/SimplerEnv/demo/simpler_data.json"
    )
    parser.add_argument("--cuda_idx", type=str, default="3")
    parser.add_argument("--api_url_file", type=str, default="demo/api_url.json")
    parser.add_argument("--output_dir", type=str, default="output/rdt")
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    main(args)
