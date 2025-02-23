from flask import Flask, request, jsonify
import argparse
from scripts.afford_inference_demo_env_minghao import (
    make_policy,
    inference_fn,
    get_arguments,
)

app = Flask(__name__)

# Load the model only once
model_path = "configs/base.yaml"
pretrained_model_name_or_path = (
    "/mnt/data/xurongtao/checkpoints/rdt-finetune-1b-afford_real_augment_qwen_warmup/checkpoint-52000/"
)

policy = make_policy(model_path, pretrained_model_name_or_path)


@app.route("/inference", methods=["POST"])
def inference():
    try:
        data = request.json

        # Create a new argument object from received data
        class Args:
            def __init__(self, data):
                self.instruction = data.get("instruction")
                self.image_path = data.get("image_path")
                self.image_previous_path = data.get("image_previous_path", None)
                self.depth_path = data.get("depth_path", None)
                self.depth_previous_path = data.get("depth_previous_path", None)
                self.pretrained_model_name_or_path = pretrained_model_name_or_path

        args = Args(data)

        # Run inference
        normalized_points, points = inference_fn(policy, args)

        return jsonify({"normalized_points": normalized_points.tolist(), "points": points.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="rdt port", default=5003, required=False)
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=5003, debug=True)
