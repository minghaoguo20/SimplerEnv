import os
import json
import random
import requests
import cv2
import numpy as np

# API URL
API_URL = "http://127.0.0.1:5003/infer"

# Temporary data directory
TEMP_DIR = "/home/xurongtao/minghao/SimplerEnv/temp_test/data_temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# Generate random image and depth data
def generate_random_image(path, width=640, height=512):
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(path, image)


# Generate test files
image_path = os.path.join(TEMP_DIR, "image.jpg")
image_previous_path = os.path.join(TEMP_DIR, "image_previous.jpg")
depth_path = os.path.join(TEMP_DIR, "depth.png")
depth_previous_path = os.path.join(TEMP_DIR, "depth_previous.png")

generate_random_image(image_path)
generate_random_image(image_previous_path)
generate_random_image(depth_path)
generate_random_image(depth_previous_path)

# Create test payload
payload = {
    "instruction": "open the door",
    "image_path": image_path,
    "image_previous_path": image_previous_path,
    "depth_path": depth_path,
    "depth_previous_path": depth_previous_path,
}

# Send request
response = requests.post(API_URL, json=payload)

# Print response
if response.status_code == 200:
    print("Inference Result:", response.json())
else:
    print("Error:", response.status_code, response.text)
