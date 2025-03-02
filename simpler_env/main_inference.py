import os

import numpy as np
import tensorflow as tf
import keras

tf.keras = keras

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator
from simpler_env.policies.octo.octo_server_model import OctoServerInference
from simpler_env.policies.rt1.rt1_model import RT1Inference

try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)

from demo.rdt_util import visualizer
from debug_util import setup_debugger
import json


if __name__ == "__main__":
    setup_debugger(ip_addr="127.0.0.1", port=9501, debug=False)
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif "a0" in args.policy_model:
        model = "a0"
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    _success_arr = np.array(success_arr, dtype=int)
    results = {"success_arr": _success_arr.tolist(), "average_success": np.mean(_success_arr)}
    results_path = os.path.join(args.logging_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    # print(f">>> Results saved to {results_path}")
    visualizer.print_note_section(f">>> Results saved to {results_path}", symbol="@")
