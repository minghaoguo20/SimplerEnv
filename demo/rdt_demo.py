# CUDA_VISIBLE_DEVICES=3 python demo/rdt_demo.py --task_names google_robot_pick_coke_can --repeat_n 1
# CUDA_VISIBLE_DEVICES=3 python demo/rdt_demo.py --task_names google_robot_pick_coke_can google_robot_move_near --repeat_n 2

import os
import json
import site
import argparse
from collections import deque
from datetime import datetime
from types import SimpleNamespace
import warnings

import numpy as np
from transformers import pipeline
from transformers import DPTImageProcessor, DPTForDepthEstimation
from scipy.spatial.transform import Rotation
import mediapy
from PIL import Image, ImageDraw, ImageFont

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien

from rdt_util import (
    create_point_cloud,
    depth_api,
    obs2pcd,
    ram_api,
    rdt_api,
    save_images_temp,
    print_progress,
    get_camera_name,
    timer,
    visualizer,
    coordination_transform,
    sim_util,
    hyperparams,
    KeyPoint,
)
from debug_util import setup_debugger

if __name__ == "__main__":
    setup_debugger(ip_addr="127.0.0.1", port=9501, debug=False)

warnings.filterwarnings("ignore")

date_now = datetime.now()
run_date = date_now.strftime("%Y%m%d_%H%M%S")
# DEPTH_SCALE = 1
DIGITS = 3
FONT_SIZE = 14
font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)


def _main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_idx
    site.main()

    task_info = run_date + "_" + args.note if len(args.note) > 0 else run_date
    task_name = args.task_name

    if "env" in locals():
        print("Closing existing env")
        env.close()
        del env
    env = simpler_env.make(task_name)
    env._max_episode_steps = 80

    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print()
    print("Reset info: ", reset_info)
    print("Instruction:", instruction)

    try:
        actors = SimpleNamespace(obj=env.unwrapped.obj, tcp=env.unwrapped.tcp)
    except:
        actors = SimpleNamespace(tcp=env.unwrapped.tcp)
    # robot = env.unwrapped.agent.robot  # 获取机器人
    # robot_link_gripper_tcp_at_world = sim_util.get_link(robot.get_links(), "link_gripper_tcp")
    # scene: sapien.Scene = env.unwrapped._scene

    save_video = SimpleNamespace(w_annotation=[], origianl=[])
    frames_save_dir = os.path.join(args.output_dir, f"{task_info}_{task_name}")
    os.makedirs(frames_save_dir, exist_ok=True)
    # print(f">>> Saving frames to: {frames_save_dir}")
    # print(f">>> TimeStamp: {date_now.strftime('%H%M%S')}")

    done, truncated = False, False

    prev_info = SimpleNamespace(
        prev_image=None,  # Initialize previous image variable
        prev_depth=None,  # Initialize previous image variable
        tcp_last_pose=np.array([-1, -1, -1]),
    )

    current_step = 0
    pose_action_pose = {}

    key_points = SimpleNamespace(
        first_a0=KeyPoint(),
        post_a0=KeyPoint(),
        pre=KeyPoint(p3d=sapien.Pose()),
        first=KeyPoint(p3d=sapien.Pose()),
        post=KeyPoint(p3d=sapien.Pose()),
    )

    # [todo] modeify here
    # quaternion = np.array(hyperparams.get_hyper(task_name, "quaternion", args.simpler_data_file))
    quaternion = actors.tcp.pose.q  # w, x, y, z
    key_points.pre.p3d.set_q(quaternion)
    key_points.first.p3d.set_q(quaternion)
    key_points.post.p3d.set_q(quaternion)

    cameras = env.unwrapped._scene.get_cameras()
    camera = cameras[1]

    camera_intrinsic = obs["camera_param"][get_camera_name(env)]["intrinsic_cv"]
    camera_extrinsic = obs["camera_param"][get_camera_name(env)]["extrinsic_cv"]
    # camera_extrinsic = obs["camera_param"][get_camera_name(env)]["cam2world_gl"]

    # ############################## point cloud pose ##############################
    points_position_world = sim_util.get_pcd_positions(camera)
    pcd_points = []
    for point in points_position_world:
        pixel = coordination_transform.project_to_image(
            coordination_transform.position_to_camera(point, camera_extrinsic), camera_intrinsic
        )
        pd = KeyPoint(p2d=pixel[::1], p3d=sapien.Pose())
        pd.p3d.set_p(point)
        pcd_points.append(pd)  # W, H
    # ##############################################################################

    # ############################## image preprocess ##############################
    image = get_image_from_maniskill2_obs_dict(env, obs)
    image_original = image.copy()
    # depth = None
    depth = sim_util.get_depth(env.unwrapped._scene, camera)  # use depth from scene
    # depth = depth_api(image=image, api_url_file=args.api_url_file) # use api
    # ##############################################################################

    # ##################################### RDT ####################################
    if prev_info.prev_image is None:
        prev_info.prev_image = image
    if prev_info.prev_depth is None:
        prev_info.prev_depth = depth
    paths = save_images_temp(image_list=[image, prev_info.prev_image, depth, prev_info.prev_depth])

    # Use prev_info.prev_image for your action model if needed
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
        raise e

    for pathh in paths:
        if pathh is not None:
            os.remove(pathh)

    key_points.first_a0.p2d = np.array(points[0])[::-1]
    key_points.post_a0.p2d = np.array(points[-1])[::-1]
    # ###############################################################################

    # ############################# find nearest object #############################
    """
    # dist_array = []
    # # all_actors = set()
    # # for atcl in env.get_articulations():
    # #     for joint in atcl.get_joints():
    # #         all_actors.add(joint)  # set 自动去重
    # all_actors = env.unwrapped._scene.get_all_actors()
    # for idx, ac in enumerate(all_actors):
    #     # print(coordination_transform.pose_to_transformation_matrix(ac.pose))
    #     p_2d = coordination_transform.project_to_image(
    #         ac.pose.p,
    #         obs["camera_param"][get_camera_name(env)]["intrinsic_cv"],
    #     )
    #     dist = coordination_transform.dist_2d(key_points.first_a0.p2d, p_2d) if (p_2d is not None) else np.inf

    #     dist_array.append(dist)
    # min_index = np.argmin(dist_array)
    # print(f"All actors: {[ac.name for ac in all_actors]}")
    # print(f"Nearest object: {all_actors[min_index].name}")
    # key_points.first.p3d.set_p(all_actors[min_index].pose.p)
    """
    nearest_point = min(
        pcd_points, key=lambda point: coordination_transform.dist_2d(key_points.first_a0.p2d, point.p2d)
    )
    key_points.first.p3d.set_p(nearest_point.p3d.p)

    _post_from_file = hyperparams.get_hyper(task_name, "post_position", args.simpler_data_file)
    if _post_from_file is not None:
        nearest_point.p3d.set_p(_post_from_file)
    else:
        nearest_point = min(
            pcd_points, key=lambda point: coordination_transform.dist_2d(key_points.post_a0.p2d, point.p2d)
        )
        key_points.post.p3d.set_p(nearest_point.p3d.p)
    # ###############################################################################

    # ################################## key point ##################################
    key_points.pre.p3d = coordination_transform.compute_pre_pose(key_points.first.p3d.p, actors.tcp.pose.q)
    # _position = hyperparams.get_hyper(task_name, "position", args.simpler_data_file)
    # if _position is not None:
    #     key_points.post.p3d.set_p(_position)
    key_points.pre.p3d_to_matrix()
    key_points.first.p3d_to_matrix()
    key_points.post.p3d_to_matrix()
    # ###############################################################################

    """
    # ########################### coordination convertion ###########################
    # # camera_at_world = obs["camera_param"][get_camera_name(env)]["cam2world_gl"]
    # camera_at_world = obs["camera_param"][get_camera_name(env)]["extrinsic_cv"]
    # robot_base_at_world = coordination_transform.pose_to_transformation_matrix(
    #     np.concatenate((robot.pose.p, robot.pose.q), axis=0)
    # )

    # # object_pose_camera = ram_result["position"]  # t_obj at camera frame [x, y, z]
    # object_pose_camera = None
    # object_pose_world = camera_at_world[:3, :3] @ object_pose_camera + camera_at_world[:3, 3]  # t_obj at world frame
    # # object_pose_world = actors.obj.get_pose().p  # from GT
    # object_rotation_world = actors.obj.get_pose().to_transformation_matrix()[:3, :3]  # from GT
    # # object_rotation_world = hyperparams.get_rotation(task_name=task_name, data_file=args.simpler_data_file)
    # object_transformation_matrix_world = np.eye(4)
    # object_transformation_matrix_world[:3, 3] = object_pose_world
    # object_transformation_matrix_world[:3, :3] = object_rotation_world
    # ################################################################################
    """

    # ################################## def stages ##################################
    stage = deque(["pre", "first", "post"])

    def get_point_and_grasp(current_stage, key_points=key_points):
        if current_stage == "pre":
            object_point = key_points.pre
            grasp_state = SimpleNamespace(start=0, end=0)
        elif current_stage == "first":
            object_point = key_points.first
            grasp_state = SimpleNamespace(start=0, end=1)
        elif current_stage == "post":
            object_point = key_points.post
            grasp_state = SimpleNamespace(start=1, end=1)
        return object_point, grasp_state

    current_stage = stage[0]
    object_point, grasp_state = get_point_and_grasp(current_stage, key_points)
    # ###############################################################################

    while not (done or truncated) and len(stage) > 0:
        """
        # action[:3]: delta xyz;
        # action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)

        # ##################################### RAM #####################################
        # pcd = obs2pcd(obs, depth_scale=DEPTH_SCALE, camera=get_camera_name(env))  # depth_scale = 2
        # ram_result = ram_api(
        #     rgb=obs["image"][get_camera_name(env)]["rgb"],
        #     pcd=pcd,
        #     contact_point=key_points.first_a0.p2d,
        #     post_contact_dir=key_points.post_a0.p2d,
        #     api_url_file=args.api_url_file,
        # )  # [x, y, z]  # it was supposed to be: [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

        #           depth_scale=0.5     | xyz=[0, 0, 0]
        #           depth_scale=1       | xyz=[0, 0, 0]
        #           depth_scale=5       | xyz=[1, 0, 2]
        #           depth_scale=10      | xyz=[2, 0, 4]
        #           depth_scale=20      | xyz=[4, 1, 9]

        # ###############################################################################

        # ################################ depth to pose ################################
        # # gt_2d = coordination_transform.project_to_image(coordination_transform.world_to_camera(actors.obj.get_pose().to_transformation_matrix(), obs["camera_param"][get_camera_name(env)]["extrinsic_cv"]), obs["camera_param"][get_camera_name(env)]["intrinsic_cv"])

        # u, v = key_points.first_a0.p2d  # 像素坐标 (x, y)
        # # u, v = gt_2d.astype(np.uint16)
        # Z_c = depth[v, u]  # 深度值
        # K = obs["camera_param"][get_camera_name(env)]["intrinsic_cv"]
        # fx, fy = K[0, 0], K[1, 1]
        # cx, cy = K[0, 2], K[1, 2]
        # # 计算相机坐标系中的3D点
        # X_c = (u - cx) * Z_c / fx
        # Y_c = (v - cy) * Z_c / fy

        # P_c = np.array([X_c, Y_c, Z_c, 1])  # 齐次坐标

        # # 获取相机到世界的变换矩阵
        # T_cam2world = obs["camera_param"][get_camera_name(env)]["cam2world_gl"]
        # # T_cam2world = obs["camera_param"][get_camera_name(env)]['extrinsic_cv']

        # # 计算世界坐标
        # P_w = T_cam2world @ P_c  # 矩阵乘法
        # X_w, Y_w, Z_w = P_w[:3]  # 取前三个元素（世界坐标）
        # object_pose_camera = P_w[:3]
        # ###############################################################################
        """

        image_info = ""

        # ########################### move to point ###########################
        action = np.zeros(7)
        gripper_transformation_matrix_world = actors.tcp.pose.to_transformation_matrix()
        object_transformation_matrix_world = object_point.p3d.to_transformation_matrix()
        action[0:6] = coordination_transform.compute_action(
            gripper_transformation_matrix_world,
            object_transformation_matrix_world,
        )
        action[6] = grasp_state.start

        offset = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float16)
        coeffi = np.array([-1, -1, 1, -1, -1, -1, 1], dtype=np.float16)
        p_magnitude = 1
        q_magnitude = 0.3 if current_stage == "pre" else 0
        coeffi[0:3] = coeffi[0:3] * p_magnitude
        coeffi[3:6] = coeffi[3:6] * q_magnitude
        action = (action - offset) * coeffi

        dist_points = coordination_transform.cal_distance(
            object_transformation_matrix_world,
            gripper_transformation_matrix_world,
        )
        angle_degrees = coordination_transform.cal_angle(
            object_transformation_matrix_world,
            gripper_transformation_matrix_world,
        )
        angle_degrees_within_limit = angle_degrees < 2 if current_stage == "pre" else True
        if dist_points < 0.02 and angle_degrees_within_limit:
            action[6] = grasp_state.end
            stage.popleft()
            if len(stage) > 0:
                current_stage = stage[0]
                object_point, grasp_state = get_point_and_grasp(current_stage, key_points)
            else:
                done = True
        # #####################################################################

        # ############################## Track & Info ##############################
        # Convert to camera coordinates
        object_camera_pos = coordination_transform.world_to_camera(object_transformation_matrix_world, camera_extrinsic)
        gripper_camera_pos = coordination_transform.world_to_camera(
            gripper_transformation_matrix_world, camera_extrinsic
        )

        # Project to 2D
        object_2d = coordination_transform.project_to_image(object_camera_pos, camera_intrinsic)
        gripper_2d = coordination_transform.project_to_image(gripper_camera_pos, camera_intrinsic)

        # Draw the projected points on the image
        # Print object_2d and gripper_2d on the image
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        draw_w_offset = SimpleNamespace(
            gt=[-100, 10],
            track=[-200, 0],
            gripper=[-100, -50],
        )

        try:
            gt_2d = coordination_transform.project_to_image(
                coordination_transform.world_to_camera(
                    actors.obj.get_pose().to_transformation_matrix(), camera_extrinsic
                ),
                camera_intrinsic,
            )
        except:
            gt_2d = None
        if gt_2d is not None:
            _object_2d = visualizer.nparray_to_string(gt_2d, DIGITS)
            _object_camera_pos = visualizer.nparray_to_string(object_camera_pos, DIGITS)
            _object_world_pos = visualizer.nparray_to_string(object_transformation_matrix_world[:3, 3], DIGITS)
            draw.ellipse([gt_2d[0] - 5, gt_2d[1] - 5, gt_2d[0] + 5, gt_2d[1] + 5], fill="yellow")
            draw.text(
                (gt_2d[0] + draw_w_offset.gt[0], gt_2d[1] + draw_w_offset.gt[1]),
                f"GT: {_object_2d}",
                fill="yellow",
                font=font,
            )
            draw.text(
                (gt_2d[0] + draw_w_offset.gt[0], gt_2d[1] + draw_w_offset.gt[1] + FONT_SIZE),
                f"Camera: {_object_camera_pos}",
                fill="yellow",
                font=font,
            )
            draw.text(
                (gt_2d[0] + draw_w_offset.gt[0], gt_2d[1] + draw_w_offset.gt[1] + 2 * FONT_SIZE),
                f"World: {_object_world_pos}",
                fill="yellow",
                font=font,
            )
        if object_2d is not None:
            _object_2d = visualizer.nparray_to_string(object_2d, DIGITS)
            _object_camera_pos = visualizer.nparray_to_string(object_camera_pos, DIGITS)
            _object_world_pos = visualizer.nparray_to_string(object_transformation_matrix_world[:3, 3], DIGITS)
            draw.ellipse([object_2d[0] - 5, object_2d[1] - 5, object_2d[0] + 5, object_2d[1] + 5], fill="blue")
            draw.text(
                (object_2d[0] + draw_w_offset.track[0], object_2d[1] + draw_w_offset.track[1]),
                f"Track: {_object_2d}",
                fill="blue",
                font=font,
            )
            draw.text(
                (object_2d[0] + draw_w_offset.track[0], object_2d[1] + draw_w_offset.track[1] + FONT_SIZE),
                f"Camera: {_object_camera_pos}",
                fill="blue",
                font=font,
            )
            draw.text(
                (object_2d[0] + draw_w_offset.track[0], object_2d[1] + draw_w_offset.track[1] + 2 * FONT_SIZE),
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
                (gripper_2d[0] + draw_w_offset.gripper[0], gripper_2d[1] + draw_w_offset.gripper[1]),
                f"Gripper: {_gripper_2d}",
                fill="green",
                font=font,
            )
            draw.text(
                (gripper_2d[0] + draw_w_offset.gripper[0], gripper_2d[1] + draw_w_offset.gripper[1] + FONT_SIZE),
                f"Camera: {_gripper_camera_pos}",
                fill="green",
                font=font,
            )
            draw.text(
                (gripper_2d[0] + draw_w_offset.gripper[0], gripper_2d[1] + draw_w_offset.gripper[1] + 2 * FONT_SIZE),
                f"World: {_gripper_world_pos}",
                fill="green",
                font=font,
            )
        action_text = f"Action: {visualizer.nparray_to_string(action, DIGITS)}"
        draw.text((10, 1 * FONT_SIZE), f"Stage: {current_stage}", fill="white", font=font)
        draw.text((10, 2 * FONT_SIZE), action_text, fill="white", font=font)
        draw.text((10, 3 * FONT_SIZE), f"Distance: {dist_points:.2f}", fill="white", font=font)
        draw.text((10, 4 * FONT_SIZE), f"Angle: {angle_degrees:.2f} degrees", fill="white", font=font)
        image = np.array(image_pil)
        # ############################################################################

        # ############################## visulize image ##############################
        image_info += f"RDT: {key_points.first_a0.p2d.tolist()} -> {key_points.post_a0.p2d.tolist()}\n"
        # Save the current image to frames_save_dir
        frames_are_different = not np.all(actors.tcp.pose.p == prev_info.tcp_last_pose)
        # if not frames_are_different: break
        if True:  # frames_are_different:
            image_save_path = os.path.join(frames_save_dir, f"frame_{current_step:04d}.png")
            # Write image info on the image
            image_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(image_pil)
            draw.text((10, 5 * FONT_SIZE), image_info, fill="white", font=font)

            # Draw an arrow from key_points.first_a0.p2d to key_points.post_a0.p2d
            start_point = (int(key_points.first_a0.p2d[0]), int(key_points.first_a0.p2d[1]))
            end_point = (int(key_points.post_a0.p2d[0]), int(key_points.post_a0.p2d[1]))
            # start_point = (int(key_points.first_a0.p2d[1]), int(key_points.first_a0.p2d[0]))
            # end_point = (int(key_points.post_a0.p2d[1]), int(key_points.post_a0.p2d[0]))
            draw.line([start_point, end_point], fill="red", width=3)
            draw.line([(10, 10), (50, 10)], fill="yellow", width=2)
            draw.polygon(
                [end_point, (end_point[0] - 5, end_point[1] - 5), (end_point[0] + 5, end_point[1] - 5)], fill="red"
            )
            # image_pil.save(image_save_path)
            image_info = ""

        pose_action_pose[f"{current_step:03d}"] = {
            "frames_are_different": frames_are_different,
            "info": f"{visualizer.nparray_to_string(prev_info.tcp_last_pose, DIGITS)} \t-> {visualizer.nparray_to_string(action, DIGITS)} \t-> {visualizer.nparray_to_string(actors.tcp.pose.p, DIGITS)}",
        }

        image = np.array(image_pil)
        save_video.w_annotation.append(image)
        save_video.origianl.append(image_original)
        # ##############################################################################

        prev_info.tcp_last_pose = actors.tcp.pose.p
        prev_info.prev_image = image  # Update prev_info.prev_image to current image
        prev_info.prev_depth = depth  # Update prev_info.prev_depth to current depth

        # ################################### action ###################################
        if not done:
            obs, reward, done, truncated, info = env.step(action)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        image_original = image.copy()
        # ##############################################################################

        # print_progress(
        #     f"\r[{current_stage}] {current_step} [{not np.all(actors.tcp.pose.p == prev_info.tcp_last_pose)}] \t| action: {visualizer.nparray_to_string(action, DIGITS)} \t| {visualizer.nparray_to_string(actors.tcp.pose.p, DIGITS)}\t"
        # )
        current_step += 1

    episode_stats = info.get("episode_stats", {})
    print("Episode stats", episode_stats)
    # Save the video instead of showing it
    str_done = "success" if done else "failure"
    video_save_path = os.path.join(frames_save_dir, f"{task_name}_{run_date}_{str_done}.mp4")
    video_original_save_path = os.path.join(frames_save_dir, f"{task_name}_{run_date}_{str_done}_original.mp4")
    mediapy.write_video(video_save_path, save_video.w_annotation, fps=10)
    mediapy.write_video(video_original_save_path, save_video.origianl, fps=10)
    print(f"Video saved to {frames_save_dir}")
    env.close()
    return done, frames_save_dir, instruction
    # # Save pose_action_pose to a text file
    # pose_action_save_path = os.path.join(frames_save_dir, f"{task_name}_{run_date}_pose_action_pose.txt")
    # with open(pose_action_save_path, "w") as f:
    #     for step in pose_action_pose:
    #         item = pose_action_pose[step]
    #         if item["frames_are_different"]:
    #             f.write(f'{step}\t{item["info"]}\n')
    #     f.write("\n" + "*" * 50 + "\n\n")
    #     for step in pose_action_pose:
    #         item = pose_action_pose[step]
    #         changed = "[Changed]" if item["frames_are_different"] else "[No Move]"
    #         f.write(f'{step} \t{changed} \t{item["info"]} \n')
    #     f.write("\n" + "*" * 50 + "\n\n")
    # print(f"Pose action data saved to {pose_action_save_path}")


@timer
def main(args):
    args.output_dir = os.path.join(args.output_dir, run_date)
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "metadata": {
            "time": run_date,
            "note": args.note,
            "task_names": args.task_names,
            "repeat_n": args.repeat_n,
            "current_dir": os.getcwd(),
        }
    }
    for task_name in args.task_names:
        visualizer.print_note_section(
            note=[f"Task: {task_name}", f"Repeat: {args.repeat_n}", f"Output: {args.output_dir}"]
        )
        success_arr = {}
        for exp_n in range(args.repeat_n):
            args.task_name = task_name
            try:
                done, save_dir, instruction = _main(args)
            except Exception as e:
                visualizer.print_note_section(note=[f"Error in task {task_name}", f"Error: {e}"])
                continue
            success_arr[f"{exp_n:04d}"] = {
                "done": done,
                "instruction": instruction,
                "save_dir": save_dir,
            }
        result[task_name] = {
            "success_arr": success_arr,
            "success_rate": (
                sum([1 for k, v in success_arr.items() if v["done"]]) / len(success_arr) if len(success_arr) > 0 else -1
            ),
        }
    args.result_path = os.path.join(args.output_dir, "result.json")
    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=4)

    final_info = []
    for task_name, task_result in result.items():
        if task_name == "metadata":
            continue
        # print(f"Task: {task_name} \t Success Rate: {task_result['success_rate']}")
        final_info.append(f"Task: {task_name} \t Success Rate: {task_result['success_rate']}")
    final_info.extend(["", f"Results saved to {args.result_path}"])
    visualizer.print_note_section(note=final_info)


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
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--repeat_n", type=int, default=10)
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        default=[
            "google_robot_pick_coke_can",
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            "google_robot_pick_standing_coke_can",
            "google_robot_pick_object",
            "google_robot_move_near_v0",
            "google_robot_move_near_v1",
            "google_robot_move_near",
            "google_robot_open_drawer",
            "google_robot_open_top_drawer",
            "google_robot_open_middle_drawer",
            "google_robot_open_bottom_drawer",
            "google_robot_close_drawer",
            "google_robot_close_top_drawer",
            "google_robot_close_middle_drawer",
            "google_robot_close_bottom_drawer",
            "google_robot_place_in_closed_drawer",
            "google_robot_place_in_closed_top_drawer",
            "google_robot_place_in_closed_middle_drawer",
            "google_robot_place_in_closed_bottom_drawer",
            "google_robot_place_apple_in_closed_top_drawer",
            "widowx_spoon_on_towel",
            "widowx_carrot_on_plate",
            "widowx_stack_cube",
            "widowx_put_eggplant_in_basket",
        ],
        help="choose from: 'google_robot_pick_coke_can', 'google_robot_pick_horizontal_coke_can', 'google_robot_pick_vertical_coke_can', 'google_robot_pick_standing_coke_can', 'google_robot_pick_object', 'google_robot_move_near_v0', 'google_robot_move_near_v1', 'google_robot_move_near', 'google_robot_open_drawer', 'google_robot_open_top_drawer', 'google_robot_open_middle_drawer', 'google_robot_open_bottom_drawer', 'google_robot_close_drawer', 'google_robot_close_top_drawer', 'google_robot_close_middle_drawer', 'google_robot_close_bottom_drawer', 'google_robot_place_in_closed_drawer', 'google_robot_place_in_closed_top_drawer', 'google_robot_place_in_closed_middle_drawer', 'google_robot_place_in_closed_bottom_drawer', 'google_robot_place_apple_in_closed_top_drawer', 'widowx_spoon_on_towel', 'widowx_carrot_on_plate', 'widowx_stack_cube', 'widowx_put_eggplant_in_basket'",
    )
    args = parser.parse_args()

    visualizer.print_note_section(note="RDT Demo")
    main(args)
