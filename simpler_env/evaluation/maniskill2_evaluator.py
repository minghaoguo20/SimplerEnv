"""
Evaluate a model on ManiSkill2 environment.
"""

import os

import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

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

from demo.rdt_util import (
    create_point_cloud,
    depth_api,
    obs2pcd,
    ram_api,
    rdt_api,
    rdt,
    save_images_temp,
    print_progress,
    get_camera_name,
    timer,
    key_points_op,
    visualizer,
    coordination_transform,
    sim_util,
    hyperparams,
    KeyPoint,
)


warnings.filterwarnings("ignore")

date_now = datetime.now()
run_date = date_now.strftime("%Y%m%d_%H%M%S")
# DEPTH_SCALE = 1
DIGITS = 3
FONT_SIZE = 14
font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)


def _run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./output/eval",
    simpler_data_file="/home/xurongtao/minghao/SimplerEnv/demo/simpler_data.json",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    annotated_images = []
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # # Initialize model
    # model.reset(task_description)

    timestep = 0
    success = "failure"

    # Step the environment
    while not (predicted_terminated or truncated):

        # ################################# prepare #################################
        try:
            actors = SimpleNamespace(obj=env.unwrapped.obj, tcp=env.unwrapped.tcp)
        except:
            actors = SimpleNamespace(tcp=env.unwrapped.tcp)
        
        prev_info = SimpleNamespace(
            prev_image=None,  # Initialize previous image variable
            prev_depth=None,  # Initialize previous image variable
            tcp_last_pose=np.array([-1, -1, -1]),
        )

        quaternion = hyperparams.get_hyper(env_name, "quaternion", simpler_data_file)
        if quaternion is not None:
            quaternion = np.array(quaternion)
        else:
            quaternion = actors.tcp.pose.q  # w, x, y, z

        key_points = SimpleNamespace(
            first_a0=KeyPoint(),
            post_a0=KeyPoint(),
            pre=KeyPoint(p3d=sapien.Pose()),
            first=KeyPoint(p3d=sapien.Pose()),
            post=KeyPoint(p3d=sapien.Pose()),
        )

        key_points.pre.p3d.set_q(quaternion)
        key_points.first.p3d.set_q(quaternion)
        key_points.post.p3d.set_q(quaternion)

        camera = env.unwrapped._scene.get_cameras()[1]
        camera_intrinsic = obs["camera_param"][get_camera_name(env)]["intrinsic_cv"]
        camera_extrinsic = obs["camera_param"][get_camera_name(env)]["extrinsic_cv"]
        # ##############################################################################

        #  point cloud pose ##############################
        pcd_points = sim_util.get_pcd_from_camera(env, obs)

        #  image preprocess ##############################
        image_original = image.copy()
        depth = sim_util.get_depth(env.unwrapped._scene, camera)

        #  RDT ###########################################
        points = rdt(image, depth, prev_info, task_description)
        key_points.first_a0.p2d = np.array(points[0])[::-1]
        key_points.post_a0.p2d = np.array(points[-1])[::-1]
        key_points = key_points_op.set_keypoints(key_points, pcd_points)

        #  def stages #####################################
        stage = deque(["pre", "first", "post"])
        current_stage = stage[0]
        object_point, grasp_state = key_points_op.get_point_and_grasp(current_stage, key_points)

        while not (done or truncated) and len(stage) > 0:
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

            action = key_points_op.offset_action(action, current_stage)

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
                    object_point, grasp_state = key_points_op.get_point_and_grasp(current_stage, key_points)
                else:
                    done = True

            _action = action.copy()
            action = dict(
                terminate_episode = [1] if done else 0,
                world_vector = _action[0:3],
                rot_axangle = _action[3:6],
                gripper = _action[6:7]
            )
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
            action_text = f"Action: {visualizer.nparray_to_string(_action, DIGITS)}"
            draw.text((10, 1 * FONT_SIZE), f"Stage: {current_stage}", fill="white", font=font)
            draw.text((10, 2 * FONT_SIZE), action_text, fill="white", font=font)
            draw.text((10, 3 * FONT_SIZE), f"Distance: {dist_points:.2f}", fill="white", font=font)
            draw.text((10, 4 * FONT_SIZE), f"Angle: {angle_degrees:.2f} degrees", fill="white", font=font)
            annotated_image = np.array(image_pil)
            # ############################################################################

            # ############################## visulize image ##############################
            image_info += f"RDT: {key_points.first_a0.p2d.tolist()} -> {key_points.post_a0.p2d.tolist()}\n"
            # Save the current image to frames_save_dir
            frames_are_different = not np.all(actors.tcp.pose.p == prev_info.tcp_last_pose)
            # if not frames_are_different: break
            if True:  # frames_are_different:
                # image_save_path = os.path.join(frames_save_dir, f"frame_{current_step:04d}.png")
                # Write image info on the image
                # image_pil = Image.fromarray(image)
                # draw = ImageDraw.Draw(image_pil)
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

            # pose_action_pose[f"{current_step:03d}"] = {
            #     "frames_are_different": frames_are_different,
            #     "info": f"{visualizer.nparray_to_string(prev_info.tcp_last_pose, DIGITS)} \t-> {visualizer.nparray_to_string(action, DIGITS)} \t-> {visualizer.nparray_to_string(actors.tcp.pose.p, DIGITS)}",
            # }

            # save_video.w_annotation.append(image)
            # save_video.origianl.append(image_original)
            annotated_images.append(np.array(image_pil))
            # ##############################################################################


            # ################################### action ###################################
            if not done:
                obs, reward, done, truncated, info = env.step(
                    np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
                )
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            # ##############################################################################


        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        # raw_action, action = model.step(image, task_description)
        # predicted_actions.append(raw_action)
        # predicted_terminated = bool(action["terminate_episode"][0] > 0)
        predicted_terminated = True
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        success = "success" if done else "failure"
        print("        " + ("\033[92m >>> success \033[0m" if done else "\033[91m >>> failure \033[0m"))
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)
    write_video(video_path[:-4] + "_annotated.mp4", annotated_images, fps=5)
    print(f"video saved to {video_path}")

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    # model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    eval_func = _run_maniskill2_eval_single_episode if isinstance(model, str) else run_maniskill2_eval_single_episode

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                eval_func(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(eval_func(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"
