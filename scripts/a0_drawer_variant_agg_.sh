# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth

output_dir=$1
xy_div=5
debug=""

declare -a policy_models=(
"a0-base"
)

declare -a env_names=(
OpenTopDrawerCustomInScene-v0
OpenMiddleDrawerCustomInScene-v0
OpenBottomDrawerCustomInScene-v0
CloseTopDrawerCustomInScene-v0
CloseMiddleDrawerCustomInScene-v0
CloseBottomDrawerCustomInScene-v0
)

EXTRA_ARGS="--enable-raytracing"


# base setup
scene_name=frl_apartment_stage_simple

EvalSim() {
  echo ${policy_model} ${env_name}

  python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
    --robot google_robot_static \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
    ${EXTRA_ARGS} \
    --logging-dir "${output_dir}/$(date +"%Y-%m-%d_%H-%M-%S")"  $( [[ -n "$debug" ]] && echo "$debug" );
}


for policy_model in "${policy_models[@]}"; do
  for env_name in "${env_names[@]}"; do
    EvalSim
  done
done


# backgrounds

declare -a scene_names=(
"modern_bedroom_no_roof"
"modern_office_no_roof"
)

for scene_name in "${scene_names[@]}"; do
  for policy_model in "${policy_models[@]}"; do
    for env_name in "${env_names[@]}"; do
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt"
      EvalSim
    done
  done
done


# lightings
scene_name=frl_apartment_stage_simple

for policy_model in "${policy_models[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter"
    EvalSim
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker"
    EvalSim
  done
done


# new cabinets
scene_name=frl_apartment_stage_simple

for policy_model in "${policy_models[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2"
    EvalSim
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3"
    EvalSim
  done
done
