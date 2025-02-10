from debug_util import setup_debugger

if __name__ == "__main__":
    setup_debugger(ip_addr="127.0.0.1", port=9501, debug=False)

import requests
import numpy as np
import open3d as o3d


def generate_random_pcd(num_points=10000, point_range=1.0):
    points = np.random.uniform(-point_range, point_range, size=(num_points, 3))  # Random 3D points
    colors = np.random.rand(num_points, 3)  # Random RGB colors (normalized to [0,1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# Generate a random point cloud
test_pcd = generate_random_pcd()

ram_url = "http://127.0.0.1:5000/lift_affordance"

data = {
    "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8).tolist(),
    "pcd": "/home/xurongtao/minghao/SimplerEnv/demo/temp_save/point_cloud_temp.pcd",
    "contact_point": [289, 377],
    "post_contact_dir": [2, 5],
}

response = requests.post(ram_url, json=data)

print(response.json())
