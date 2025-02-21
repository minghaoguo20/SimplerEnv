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
    print_progress,
    get_camera_name,
    visualizer,
    coordination_transform,
    sim_util,
    hyperparams,
)
import shutil
import tempfile
from datetime import datetime
import sys
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
import warnings

from debug_util import setup_debugger

if __name__ == "__main__":
    setup_debugger(ip_addr="127.0.0.1", port=9501, debug=False)

warnings.filterwarnings("ignore")

date_now = datetime.now()
run_date = date_now.strftime("%Y%m%d_%H%M%S")
# DEPTH_SCALE = 1.0067516767893736
DEPTH_SCALE = 1
DIGITS = 3
FONT_SIZE = 14  # 可以调整这个值来更改字体大小
font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)  # 指定字体文件（Windows 默认字体）


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
    env._max_episode_steps = 80
    # sapien.render_config.rt_use_denoiser = False
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print("Reset info", reset_info)
    print("Instruction", instruction)

    # env.unwrapped.obj
    # env.unwrapped.tcp
    # env.unwrapped._actors
    actors = SimpleNamespace(obj=env.unwrapped.obj, tcp=env.unwrapped.tcp, robot=env.unwrapped.agent.robot)
    robot = env.unwrapped.agent.robot  # 获取机器人
    robot_link_gripper_tcp_at_world = sim_util.get_link(robot.get_links(), "link_gripper_tcp")

    scene: sapien.Scene = env.unwrapped._scene
    # all_actors = scene.get_all_actors()

    frames = []
    frames_save_dir = os.path.join(args.output_dir, f"{task_name}_{task_info}")
    os.makedirs(frames_save_dir, exist_ok=True)
    print(f">>> Saving frames to: {frames_save_dir}")
    print(f">>> Time: {date_now.strftime('%H%M%S')}")

    done, truncated = False, False

    prev_info = SimpleNamespace(
        prev_image=None,  # Initialize previous image variable
        prev_depth=None,  # Initialize previous image variable
        tcp_last_pose=np.array([-1, -1, -1]),
    )
    retrial_times = 5

    current_step = 0
    image_info = ""
    pose_action_pose = {}
    first_point = [0, 0]
    point_direction = [0, 0]

    cameras = scene.get_cameras()
    camera = cameras[1]

    while not (done or truncated) and retrial_times > 0 and current_step < 10:
        # action[:3]: delta xyz;
        # action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)

        # ############################## image preprocess ##############################
        image = get_image_from_maniskill2_obs_dict(env, obs)
        # image = np.array(Image.fromarray(image).resize((640, 480)))
        depth = sim_util.get_depth(scene, camera)
        # depth = depth_api(image=image, api_url_file=args.api_url_file)
        # depth = np.array(Image.fromarray(depth).resize((640, 480)))
        # ##############################################################################

        # ##################################### RDT ####################################
        if prev_info.prev_image is None:
            prev_info.prev_image = image
        if prev_info.prev_depth is None:
            prev_info.prev_depth = depth
        paths = save_images_temp(image_list=[image, prev_info.prev_image, depth, prev_info.prev_depth])

        # Use prev_info.prev_image for your action model if needed
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
        # ###############################################################################

        first_point = points[0]
        point_direction = points[1]
        # point_direction = [points[1][0] - points[0][0], points[1][1] - points[0][1]]
        image_info += f"RDT: {first_point} -> {point_direction}\n"

        # ##################################### RAM #####################################
        # pcd = obs2pcd(obs, depth_scale=DEPTH_SCALE, camera=get_camera_name(env))  # depth_scale = 2
        # ram_result = ram_api(
        #     rgb=obs["image"][get_camera_name(env)]["rgb"],
        #     pcd=pcd,
        #     contact_point=first_point,
        #     post_contact_dir=point_direction,
        #     api_url_file=args.api_url_file,
        # )  # [x, y, z]  # it was supposed to be: [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        # if ram_result is None:
        #     retrial_times -= 1
        #     continue
        # ###############################################################################
        """
            depth_scale=0.5     | xyz=[0, 0, 0]
            depth_scale=1       | xyz=[0, 0, 0]
            depth_scale=5       | xyz=[1, 0, 2]
            depth_scale=10      | xyz=[2, 0, 4]
            depth_scale=20      | xyz=[4, 1, 9]
        """

        # gt_2d = coordination_transform.project_to_image(coordination_transform.world_to_camera(actors.obj.get_pose().to_transformation_matrix(), obs["camera_param"][get_camera_name(env)]["extrinsic_cv"]), obs["camera_param"][get_camera_name(env)]["intrinsic_cv"])
        depth = depth * DEPTH_SCALE

        u, v = first_point  # 像素坐标 (x, y)
        # u, v = gt_2d.astype(np.uint16)
        Z_c = depth[u, v]   # 深度值
        K = obs["camera_param"][get_camera_name(env)]['intrinsic_cv']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # 计算相机坐标系中的3D点
        X_c = (u - cx) * Z_c / fx
        Y_c = (v - cy) * Z_c / fy

        P_c = np.array([X_c, Y_c, Z_c, 1])  # 齐次坐标

        # 获取相机到世界的变换矩阵
        T_cam2world = obs["camera_param"][get_camera_name(env)]['cam2world_gl']
        # T_cam2world = obs["camera_param"][get_camera_name(env)]['extrinsic_cv']

        # 计算世界坐标
        P_w = T_cam2world @ P_c  # 矩阵乘法
        X_w, Y_w, Z_w = P_w[:3]  # 取前三个元素（世界坐标）
        object_pose_camera = P_w[:3]

        # image_info += f"GT2D -> camera frame = [{object_pose_camera}]\n"

        # ########################### coordination convertion ###########################
        action = np.zeros(7)

        # camera_at_world = obs["camera_param"][get_camera_name(env)]["cam2world_gl"]
        camera_at_world = obs["camera_param"][get_camera_name(env)]["extrinsic_cv"]
        robot_base_at_world = coordination_transform.p_q_to_transformation_matrix(
            np.concatenate((robot.pose.p, robot.pose.q), axis=0)
        )

        # object_pose_camera = ram_result["position"]  # t_obj at camera frame [x, y, z]
        object_pose_world = (
            camera_at_world[:3, :3] @ object_pose_camera + camera_at_world[:3, 3]
        )  # t_obj at world frame
        # object_pose_world = actors.obj.get_pose().p  # from GT
        object_rotation_world = actors.obj.get_pose().to_transformation_matrix()[:3, :3]  # from GT
        # object_rotation_world = hyperparams.get_rotation(task_name=task_name, data_file=args.simpler_data_file)
        object_transformation_matrix_world = np.eye(4)
        object_transformation_matrix_world[:3, 3] = object_pose_world
        object_transformation_matrix_world[:3, :3] = object_rotation_world

        gripper_transformation_matrix_world = actors.tcp.pose.to_transformation_matrix()

        action[0:6] = coordination_transform.compute_action(
            gripper_transformation_matrix_world,
            object_transformation_matrix_world,
        )
        action[6] = 0

        offset = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float16)
        coeffi = np.array([-1, -1, 1, 1, 1, 1, 1], dtype=np.float16)
        pose_magnitude = 1
        rot_magnitude = 0.2
        coeffi[0:3] = coeffi[0:3] * pose_magnitude
        coeffi[3:6] = coeffi[3:6] * rot_magnitude
        action = (action - offset) * coeffi

        dist_points = coordination_transform.cal_distance(
            object_transformation_matrix_world,
            gripper_transformation_matrix_world,
        )
        angle_degrees = coordination_transform.cal_angle(
            object_transformation_matrix_world,
            gripper_transformation_matrix_world,
        )
        if dist_points < 0.02:
            action[6] = 1
        elif dist_points > 0.1:
            action[6] = 0
        # ################################################################################

        # ############################## Track & Info ##############################
        intrinsic_matrix = obs["camera_param"][get_camera_name(env)]["intrinsic_cv"]
        camera_extrinsic = obs["camera_param"][get_camera_name(env)]["extrinsic_cv"]

        # Convert to camera coordinates
        object_camera_pos = coordination_transform.world_to_camera(object_transformation_matrix_world, camera_extrinsic)
        gripper_camera_pos = coordination_transform.world_to_camera(
            gripper_transformation_matrix_world, camera_extrinsic
        )

        # Project to 2D
        object_2d = coordination_transform.project_to_image(object_camera_pos, intrinsic_matrix)
        gripper_2d = coordination_transform.project_to_image(gripper_camera_pos, intrinsic_matrix)

        # Draw the projected points on the image
        # Print object_2d and gripper_2d on the image
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        draw_w_offset = 200

        gt_2d = coordination_transform.project_to_image(coordination_transform.world_to_camera(actors.obj.get_pose().to_transformation_matrix(), camera_extrinsic), intrinsic_matrix)
        if gt_2d is not None:
            _object_2d = visualizer.nparray_to_string(gt_2d, DIGITS)
            _object_camera_pos = visualizer.nparray_to_string(object_camera_pos, DIGITS)
            _object_world_pos = visualizer.nparray_to_string(object_transformation_matrix_world[:3, 3], DIGITS)
            draw.ellipse([gt_2d[0] - 5, gt_2d[1] - 5, gt_2d[0] + 5, gt_2d[1] + 5], fill="yellow")
            draw.text((gt_2d[0] - draw_w_offset, gt_2d[1]), f"Object: {_object_2d}", fill="yellow", font=font)
            draw.text(
                (gt_2d[0] - draw_w_offset, gt_2d[1] + FONT_SIZE),
                f"Camera: {_object_camera_pos}",
                fill="yellow",
                font=font,
            )
            draw.text(
                (gt_2d[0] - draw_w_offset, gt_2d[1] + 2 * FONT_SIZE),
                f"World: {_object_world_pos}",
                fill="yellow",
                font=font,
            )
        if object_2d is not None:
            _object_2d = visualizer.nparray_to_string(object_2d, DIGITS)
            _object_camera_pos = visualizer.nparray_to_string(object_camera_pos, DIGITS)
            _object_world_pos = visualizer.nparray_to_string(object_transformation_matrix_world[:3, 3], DIGITS)
            draw.ellipse([object_2d[0] - 5, object_2d[1] - 5, object_2d[0] + 5, object_2d[1] + 5], fill="blue")
            draw.text((object_2d[0] - draw_w_offset, object_2d[1]), f"Object: {_object_2d}", fill="blue", font=font)
            draw.text(
                (object_2d[0] - draw_w_offset, object_2d[1] + FONT_SIZE),
                f"Camera: {_object_camera_pos}",
                fill="blue",
                font=font,
            )
            draw.text(
                (object_2d[0] - draw_w_offset, object_2d[1] + 2 * FONT_SIZE),
                f"World: {_object_world_pos}",
                fill="blue",
                font=font,
            )
        if gripper_2d is not None:
            _gripper_2d = visualizer.nparray_to_string(gripper_2d, DIGITS)
            _gripper_camera_pos = visualizer.nparray_to_string(gripper_camera_pos, DIGITS)
            _gripper_world_pos = visualizer.nparray_to_string(gripper_transformation_matrix_world[:3, 3], DIGITS)
            draw.ellipse([gripper_2d[0] - 5, gripper_2d[1] - 5, gripper_2d[0] + 5, gripper_2d[1] + 5], fill="green")
            draw.text(
                (gripper_2d[0] - draw_w_offset, gripper_2d[1]), f"Gripper: {_gripper_2d}", fill="green", font=font
            )
            draw.text(
                (gripper_2d[0] - draw_w_offset, gripper_2d[1] + FONT_SIZE),
                f"Camera: {_gripper_camera_pos}",
                fill="green",
                font=font,
            )
            draw.text(
                (gripper_2d[0] - draw_w_offset, gripper_2d[1] + 2 * FONT_SIZE),
                f"World: {_gripper_world_pos}",
                fill="green",
                font=font,
            )
        action_text = f"Action: {visualizer.nparray_to_string(action, DIGITS)}"
        draw.text((10, 1 * FONT_SIZE), action_text, fill="white", font=font)
        draw.text((10, 2 * FONT_SIZE), f"Distance: {dist_points:.2f}", fill="white", font=font)
        draw.text((10, 3 * FONT_SIZE), f"Angle: {angle_degrees:.2f} degrees", fill="white", font=font)
        image = np.array(image_pil)
        # ############################################################################

        # ############################## visulize image ##############################
        # Save the current image to frames_save_dir
        frames_are_different = not np.all(actors.tcp.pose.p == prev_info.tcp_last_pose)
        # if not frames_are_different: break
        if True:  # frames_are_different:
            image_save_path = os.path.join(frames_save_dir, f"frame_{current_step:04d}.png")
            # Write image info on the image
            image_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(image_pil)
            draw.text((10, 5 * FONT_SIZE), image_info, fill="white", font=font)

            # Draw an arrow from first_point to point_direction
            start_point = (int(first_point[0]), int(first_point[1]))
            end_point = (int(point_direction[0]), int(point_direction[1]))
            draw.line([start_point, end_point], fill="red", width=3)
            draw.polygon(
                [end_point, (end_point[0] - 5, end_point[1] - 5), (end_point[0] + 5, end_point[1] - 5)], fill="red"
            )
            # image_pil.save(image_save_path)
            image_info = ""

        pose_action_pose[f"{current_step:03d}"] = {
            "frames_are_different": frames_are_different,
            "info": f"{visualizer.nparray_to_string(prev_info.tcp_last_pose, DIGITS)} \t-> {visualizer.nparray_to_string(action, DIGITS)} \t-> {visualizer.nparray_to_string(actors.tcp.pose.p, DIGITS)}",
        }
        # ##############################################################################

        image = np.array(image_pil)
        frames.append(image)

        prev_info.tcp_last_pose = actors.tcp.pose.p
        prev_info.prev_image = image  # Update prev_info.prev_image to current image
        prev_info.prev_depth = depth  # Update prev_info.prev_depth to current depth

        obs, reward, done, truncated, info = env.step(action)

        print_progress(
            f'\rStep {current_step} [{not np.all(actors.tcp.pose.p == prev_info.tcp_last_pose)}] \t| action: {visualizer.nparray_to_string(action, DIGITS)} \t| {visualizer.nparray_to_string(obs["extra"]["tcp_pose"][0:3], DIGITS)}\t'
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
                f.write(f'{step}\t{item["info"]}\n')
        f.write("\n" + "*" * 50 + "\n\n")
        for step in pose_action_pose:
            item = pose_action_pose[step]
            changed = "[Changed]" if item["frames_are_different"] else "[No Move]"
            f.write(f'{step} \t{changed} \t{item["info"]} \n')
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
