# First, open a virtual display:
#   ps aux | grep X
#   kill -9 xxxx
#   nohup sudo X :0 &
# Then, set the display environment variable:
#   export DISPLAY=:0
# Finally, run the script:
#   CUDA_VISIBLE_DEVICES=1 python demo/demo-rdt.py
from debug_util import setup_debugger

if __name__ == "__main__":
    setup_debugger(ip_addr="127.0.0.1", port=9501, debug=False)

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
from rdt_util import create_point_cloud

site.main()

task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

if "env" in locals():
    print("Closing existing env")
    env.close()
    del env
env = simpler_env.make(task_name)
# env = simpler_env.make(task_name=task_name, obs_mode="rgbd")
# Colab GPU does not supoort denoiser
sapien.render_config.rt_use_denoiser = False
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)
base_intrinsic = obs["camera_param"]["base_camera"]["intrinsic_cv"]

rgb = obs["image"]["base_camera"]["rgb"]
depth = obs["image"]["base_camera"]["depth"]
seg = obs["image"]["base_camera"]["Segmentation"][:, :, 0]
intrinsic_matrix = obs["camera_param"]["base_camera"]["intrinsic_cv"]
intrinsics_dict = {
    "fx": obs["camera_param"]["base_camera"]["intrinsic_cv"][0, 0],
    "fy": obs["camera_param"]["base_camera"]["intrinsic_cv"][1, 1],
    "ppx": obs["camera_param"]["base_camera"]["intrinsic_cv"][0, 2],
    "ppy": obs["camera_param"]["base_camera"]["intrinsic_cv"][1, 2]
}
pcd = create_point_cloud(depth_image=depth, color_image=rgb, depth_scale=1.0, intrinsics=intrinsics_dict, seg=seg) # tuple(np.array(length, 3), np.array(length, 3))

frames = []
done, truncated = False, False
while not (done or truncated):
    # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
    # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    result = depth_estimator(images=image)
    action = env.action_space.sample()  # replace this with your policy inference
    obs, reward, done, truncated, info = env.step(action)
    frames.append(image)

episode_stats = info.get("episode_stats", {})
print("Episode stats", episode_stats)
# Save the video instead of showing it
video_save_path = "output/demo-rdt.mp4"
mediapy.write_video(video_save_path, frames, fps=10)
