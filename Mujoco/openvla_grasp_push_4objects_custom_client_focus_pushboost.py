import argparse
import base64
import io
import json
import math
import os
import re
from contextlib import nullcontext
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import requests
from PIL import Image
from sshtunnel import SSHTunnelForwarder

from raccoon_env import SyncSimRaccoonEnv


CYLINDER_BODY_BY_COLOR = {
    "red": "target_object",
    "blue": "target_object_blue",
    "green": "target_object_green",
    "yellow": "target_object_yellow",
}

CUBE_BODY_BY_COLOR = {
    "red": "target_cube_red",
    "blue": "target_cube_blue",
    "green": "target_cube_green",
    "yellow": "target_cube_yellow",
}

BODY_BY_SHAPE_AND_COLOR = {
    "cylinder": CYLINDER_BODY_BY_COLOR,
    "cube": CUBE_BODY_BY_COLOR,
}

OBJECT_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())
OBJECT_SHAPES = tuple(BODY_BY_SHAPE_AND_COLOR.keys())

# Assignment setup: use only four active objects in the scene.
#   - green cylinder
#   - yellow cylinder
#   - red cube
#   - blue cube
# The XML may still contain other objects, but reset_multicolor_scene() hides unused bodies.
CUSTOM_TARGET_KEYS = (
    "green_cylinder",
    "yellow_cylinder",
    "red_cube",
    "blue_cube",
)
TARGET_KEYS = CUSTOM_TARGET_KEYS

TASK_TYPES = ("grasp", "push")
SUPPORTED_TASK_TARGETS = {
    "grasp": TARGET_KEYS,
    # Push is limited to cubes because cylinders roll unpredictably.
    "push": ("red_cube", "blue_cube"),
}

# Backward-compatible alias used by some legacy code paths.
CYLINDER_COLORS = OBJECT_COLORS

# Dataset collection code와 동일한 기본 배치 조건.
# 이전 단일 object range였던 x=(-0.18, 0.18), y=(0.10, 0.18)보다
# x는 좁게, y는 조금 더 앞으로 제한한다.
DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
DEFAULT_OBJECT_Y_RANGE = (0.16, 0.20)
DEFAULT_MIN_OBJECT_DISTANCE = 0.035
DEFAULT_YAW_RANGE = (-math.pi / 4, math.pi / 4)
DEFAULT_INSTRUCTION_TEMPLATE = "{task} the {color} {shape}"


def image_to_b64(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def request_action(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    timeout: float = 60.0,
) -> Dict[str, Any]:
    payload = {
        "instruction": instruction,
        "image_b64": image_to_b64(image_rgb),
        "unnorm_key": unnorm_key,
        "do_sample": False,
    }
    response = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    if not response.ok:
        print(f"[SERVER ERROR] {response.status_code} | {response.text}")
        response.raise_for_status()
    return response.json()


def resolve_ssh_password(args: argparse.Namespace) -> Optional[str]:
    if args.ssh_password:
        return args.ssh_password
    env_password = os.environ.get("OPENVLA_SSH_PASSWORD")
    if env_password:
        return env_password
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def open_ssh_tunnel(args: argparse.Namespace) -> SSHTunnelForwarder:
    ssh_password = resolve_ssh_password(args)
    tunnel = SSHTunnelForwarder(
        ssh_address_or_host=(args.ssh_host, args.ssh_port),
        ssh_username=args.ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=(args.remote_server_host, args.remote_server_port),
        local_bind_address=(args.local_server_host, args.local_server_port),
    )
    tunnel.start()
    return tunnel


def build_server_url(args: argparse.Namespace, tunnel: Optional[SSHTunnelForwarder]) -> str:
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required when --use_ssh_tunnel is not enabled.")
    return args.server_url


def maybe_tunnel_context(args: argparse.Namespace):
    if args.use_ssh_tunnel:
        return open_ssh_tunnel(args)
    return nullcontext(None)


def print_success_log(step_idx: int, exec_info: Dict[str, Any]) -> None:
    final_delta_xyz = [round(float(v), 4) for v in exec_info["final_delta_xyz"]]
    move_xyz = [round(float(v), 4) for v in exec_info["actual_move_xyz"]]
    target_xyz = [round(float(v), 4) for v in exec_info["target_xyz"]]
    gripper = float(exec_info["gripper_cmd"])
    retries = int(exec_info["retry_count"])
    print(
        f"[{step_idx:03d}] OK | final_delta={final_delta_xyz} | "
        f"move={move_xyz} | target={target_xyz} | "
        f"gripper={gripper:.1f} | retries={retries}"
    )


def print_fail_log(step_idx: int, exc: Exception) -> None:
    print(f"[{step_idx:03d}] FAIL | {exc}")


def infer_color_from_instruction(instruction: Optional[str]) -> Optional[str]:
    """Return the single color word found in an instruction, or None."""
    if not instruction:
        return None

    text = instruction.lower()
    matches = []
    for color in OBJECT_COLORS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            matches.append(color)

    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 색상이 들어 있습니다: {matches} | instruction={instruction!r}")
    return matches[0] if matches else None


def infer_shape_from_instruction(instruction: Optional[str]) -> Optional[str]:
    """Return the single shape word found in an instruction, or None."""
    if not instruction:
        return None

    text = instruction.lower()
    matches = []
    for shape in OBJECT_SHAPES:
        if re.search(rf"\b{re.escape(shape)}\b", text):
            matches.append(shape)

    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 모양이 들어 있습니다: {matches} | instruction={instruction!r}")
    return matches[0] if matches else None


def infer_task_from_instruction(instruction: Optional[str]) -> Optional[str]:
    """Return the single task word found in an instruction, or None."""
    if not instruction:
        return None

    text = instruction.lower()
    matches = []
    for task in TASK_TYPES:
        if re.search(rf"\b{re.escape(task)}\b", text):
            matches.append(task)

    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 task가 들어 있습니다: {matches} | instruction={instruction!r}")
    return matches[0] if matches else None


def make_target_key(color: str, shape: str) -> str:
    return f"{color}_{shape}"


def split_target_key(target_key: str) -> Tuple[str, str]:
    color, shape = target_key.split("_", 1)
    return color, shape


def resolve_target_and_instruction(
    instruction: Optional[str],
    target_color_arg: str,
    target_shape_arg: str,
    task_type_arg: str,
    rng: np.random.Generator,
    instruction_template: str,
) -> Tuple[str, str, str, str, str]:
    """
    Keep the OpenVLA prompt and the physical target synchronized.

    Priority:
      1. If instruction contains task/color/shape, use them.
      2. Else use --task_type / --target_color / --target_shape when provided.
      3. Else randomize missing fields from the supported 4-object setup.
    """
    instruction_task = infer_task_from_instruction(instruction)
    instruction_color = infer_color_from_instruction(instruction)
    instruction_shape = infer_shape_from_instruction(instruction)

    if instruction_task is not None:
        task_type = instruction_task
        if task_type_arg in TASK_TYPES and task_type_arg != instruction_task:
            raise ValueError(
                f"--instruction task({instruction_task})와 --task_type({task_type_arg})가 다릅니다. "
                "OpenVLA prompt와 실제 target이 어긋나지 않도록 둘 중 하나를 수정하세요."
            )
    elif task_type_arg in TASK_TYPES:
        task_type = task_type_arg
    elif task_type_arg in ("auto", "random"):
        task_type = str(rng.choice(TASK_TYPES))
    else:
        raise ValueError(f"지원하지 않는 --task_type 값입니다: {task_type_arg}")

    supported_target_keys = tuple(SUPPORTED_TASK_TARGETS[task_type])

    if instruction_color is not None:
        target_color = instruction_color
        if target_color_arg in OBJECT_COLORS and target_color_arg != instruction_color:
            raise ValueError(
                f"--instruction 색상({instruction_color})과 --target_color({target_color_arg})가 다릅니다. "
                "OpenVLA prompt와 실제 target이 어긋나지 않도록 둘 중 하나를 수정하세요."
            )
    elif target_color_arg in OBJECT_COLORS:
        target_color = target_color_arg
    elif target_color_arg in ("auto", "random"):
        target_color = str(rng.choice([split_target_key(k)[0] for k in supported_target_keys]))
    else:
        raise ValueError(f"지원하지 않는 --target_color 값입니다: {target_color_arg}")

    if instruction_shape is not None:
        target_shape = instruction_shape
        if target_shape_arg in OBJECT_SHAPES and target_shape_arg != instruction_shape:
            raise ValueError(
                f"--instruction 모양({instruction_shape})과 --target_shape({target_shape_arg})가 다릅니다. "
                "OpenVLA prompt와 실제 target이 어긋나지 않도록 둘 중 하나를 수정하세요."
            )
    elif target_shape_arg in OBJECT_SHAPES:
        target_shape = target_shape_arg
    elif target_shape_arg in ("auto", "random"):
        candidates = [k for k in supported_target_keys if split_target_key(k)[0] == target_color]
        if len(candidates) == 0:
            candidates = list(supported_target_keys)
        target_shape = str(rng.choice([split_target_key(k)[1] for k in candidates]))
    else:
        raise ValueError(f"지원하지 않는 --target_shape 값입니다: {target_shape_arg}")

    target_key = make_target_key(target_color, target_shape)
    if target_key not in supported_target_keys:
        raise ValueError(
            f"현재 설정에서는 task={task_type!r}, target={target_key!r} 조합을 지원하지 않습니다. "
            f"지원 조합: {[(task_type, k) for k in supported_target_keys]}"
        )

    if instruction is None or instruction.strip() == "":
        instruction = instruction_template.format(
            task=task_type,
            color=target_color,
            shape=target_shape,
        )

    return task_type, target_key, target_color, target_shape, instruction

def make_default_object_specs() -> Dict[str, Dict[str, float]]:
    """Deterministic fallback used when randomization is disabled."""
    x_values = np.linspace(
        DEFAULT_OBJECT_X_RANGE[0] * 0.75,
        DEFAULT_OBJECT_X_RANGE[1] * 0.75,
        len(TARGET_KEYS),
    )
    y_center = float(sum(DEFAULT_OBJECT_Y_RANGE) / 2.0)

    specs: Dict[str, Dict[str, float]] = {}
    for idx, target_key in enumerate(TARGET_KEYS):
        color, shape = split_target_key(target_key)
        specs[target_key] = {
            "body_name": BODY_BY_SHAPE_AND_COLOR[shape][color],
            "color": color,
            "shape": shape,
            "x": float(x_values[idx]),
            "y": y_center,
            "yaw": 0.0,
        }
    return specs


def focus_push_target_scene(
    object_specs: Dict[str, Dict[str, float]],
    target_key: str,
) -> Dict[str, Dict[str, float]]:
    """
    Push demo 안정화용 배치 보정.

    - push target cube를 화면/작업공간 중앙 근처에 둔다.
    - non-target cube와 cylinders는 뒤쪽/옆쪽으로 보내서 색상 혼동을 줄인다.
    - y를 0.155로 낮춰서 y=0.20 workspace limit까지 더 많이 밀 공간을 확보한다.
    """
    object_specs = {k: dict(v) for k, v in object_specs.items()}

    # 기본 distractor 배치
    fixed_layout = {
        "green_cylinder": (-0.080, 0.190, 0.0),
        "yellow_cylinder": (0.080, 0.190, 0.0),
        "red_cube": (-0.085, 0.175, 0.0),
        "blue_cube": (0.085, 0.175, 0.0),
    }

    for key, (x, y, yaw) in fixed_layout.items():
        if key in object_specs:
            object_specs[key]["x"] = float(x)
            object_specs[key]["y"] = float(y)
            object_specs[key]["yaw"] = float(yaw)

    # target cube는 중앙/앞쪽에 고정
    if target_key in object_specs:
        object_specs[target_key]["x"] = 0.0
        object_specs[target_key]["y"] = 0.155
        object_specs[target_key]["yaw"] = 0.0

    # 반대 색 cube는 뒤쪽 옆으로 보내서 target과 덜 헷갈리게 함
    if target_key == "red_cube" and "blue_cube" in object_specs:
        object_specs["blue_cube"]["x"] = 0.095
        object_specs["blue_cube"]["y"] = 0.190
        object_specs["blue_cube"]["yaw"] = 0.0
    elif target_key == "blue_cube" and "red_cube" in object_specs:
        object_specs["red_cube"]["x"] = -0.095
        object_specs["red_cube"]["y"] = 0.190
        object_specs["red_cube"]["yaw"] = 0.0

    return object_specs


def sample_object_specs(
    rng: np.random.Generator,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    yaw_range: Tuple[float, float] = DEFAULT_YAW_RANGE,
    min_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    max_tries: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Dataset collection code와 동일한 조건으로 cylinder 4개 + cube 4개를 모두 배치한다.
    """
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

    specs: Dict[str, Dict[str, float]] = {}
    placed_xy = []

    # 특정 target이 항상 먼저 배치되어 유리/불리해지는 bias를 줄인다.
    placement_order = list(TARGET_KEYS)
    rng.shuffle(placement_order)

    for target_key in placement_order:
        color, shape = split_target_key(target_key)
        body_name = BODY_BY_SHAPE_AND_COLOR[shape][color]

        for _ in range(max_tries):
            x = float(rng.uniform(x_range[0], x_range[1]))
            y = float(rng.uniform(y_range[0], y_range[1]))
            xy = np.array([x, y], dtype=np.float64)

            if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                specs[target_key] = {
                    "body_name": body_name,
                    "color": color,
                    "shape": shape,
                    "x": x,
                    "y": y,
                    "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                }
                placed_xy.append(xy)
                break
        else:
            raise RuntimeError(
                "cylinder 4개 + cube 4개를 겹치지 않게 배치하지 못했습니다. "
                f"x_range={x_range}, y_range={y_range}, min_distance={min_distance}를 확인하세요."
            )

    return {target_key: specs[target_key] for target_key in TARGET_KEYS}

def reset_freejoint_body_pose(env: SyncSimRaccoonEnv, body_name: str, x: float, y: float, z: float, yaw: float) -> None:
    """Set a MuJoCo freejoint body pose directly through env.model/env.data."""
    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise AttributeError("SyncSimRaccoonEnv에 model/data 속성이 필요합니다.")

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"body not found: {body_name}. XML이 Raccoon_colored_cylinder.xml인지 확인하세요.")

    jnt_adr = int(env.model.body_jntadr[body_id])
    jnt_num = int(env.model.body_jntnum[body_id])
    if jnt_num < 1:
        raise ValueError(f"{body_name} has no joint")

    joint_id = jnt_adr
    qpos_adr = int(env.model.jnt_qposadr[joint_id])

    # freejoint qpos = [x, y, z, qw, qx, qy, qz]
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    env.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

    qvel_adr = int(env.model.jnt_dofadr[joint_id])
    env.data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def reset_multicolor_scene(
    env: SyncSimRaccoonEnv,
    object_specs: Dict[str, Dict[str, float]],
    target_key: str,
) -> None:
    """
    Reset the robot using the existing env.reset_episode(), then place all colored
    cylinders and cubes in the scene. The prompted target is stored as env.active_object_body_name
    when the env supports that attribute, but inference only needs the rendered image.
    """
    if target_key not in object_specs:
        raise ValueError(f"target_key={target_key}가 object_specs에 없습니다.")

    target_spec = object_specs[target_key]

    # Existing raccoon_env expects a single target pose for reset_episode().
    # We use the prompted target pose to reset the robot/home state, then override
    # all object poses below.
    env.reset_episode(float(target_spec["x"]), float(target_spec["y"]), float(target_spec["yaw"]))

    for _, spec in object_specs.items():
        reset_freejoint_body_pose(
            env=env,
            body_name=str(spec["body_name"]),
            x=float(spec["x"]),
            y=float(spec["y"]),
            z=0.02,
            yaw=float(spec["yaw"]),
        )

    # Hide unused bodies that still exist in the XML.
    # This keeps the active scene to only the selected four objects.
    active_body_names = {str(spec["body_name"]) for spec in object_specs.values()}
    for shape, body_by_color in BODY_BY_SHAPE_AND_COLOR.items():
        for color, body_name in body_by_color.items():
            if body_name in active_body_names:
                continue
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue
            reset_freejoint_body_pose(
                env=env,
                body_name=body_name,
                x=10.0,
                y=10.0,
                z=0.02,
                yaw=0.0,
            )

    target_body_name = str(target_spec["body_name"])
    if hasattr(env, "active_object_body_name"):
        env.active_object_body_name = target_body_name
    if hasattr(env, "target_body_name"):
        env.target_body_name = target_body_name

    mujoco.mj_forward(env.model, env.data)

def object_specs_to_meta(object_specs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    return {
        target_key: {
            "body_name": str(spec["body_name"]),
            "color": str(spec.get("color", "")),
            "shape": str(spec.get("shape", "")),
            "xy": [float(spec["x"]), float(spec["y"])],
            "yaw": float(spec["yaw"]),
        }
        for target_key, spec in object_specs.items()
    }

def write_rollout_meta(
    out_dir: Path,
    instruction: str,
    task_type: str,
    target_key: str,
    target_color: str,
    target_shape: str,
    object_specs: Dict[str, Dict[str, float]],
    args: Dict[str, Any],
) -> None:
    meta = {
        "instruction": instruction,
        "task_type": task_type,
        "target_key": target_key,
        "target_color": target_color,
        "target_shape": target_shape,
        "target_body_name": object_specs[target_key]["body_name"],
        "all_object_init_poses": object_specs_to_meta(object_specs),
        "args": args,
    }
    with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def rollout(
    xml_path: str,
    server_url: str,
    instruction: Optional[str],
    unnorm_key: str,
    output_dir: str,
    episode_id: int = 1,
    max_steps: int = 1000000,
    use_viewer: bool = True,
    camera_name: str = "front_view",
    speed: int = 70,
    settle_seconds_per_action: float = 0.8,
    initial_settle_seconds: float = 0.3,
    delta_scale: float = 1.0,
    randomize_objects: bool = True,
    request_timeout: float = 60.0,
    max_delta_xyz: float = 0.005,
    target_color_arg: str = "auto",
    target_shape_arg: str = "auto",
    task_type_arg: str = "auto",
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    seed: Optional[int] = None,
    object_x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    object_y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    min_object_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
) -> None:
    out_dir = Path(output_dir) / f"episode_{episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 기존 이미지 삭제 후 새로 저장 시작
    clear_existing_images(out_dir)

    rng = np.random.default_rng(seed)
    task_type, target_key, target_color, target_shape, instruction = resolve_target_and_instruction(
        instruction=instruction,
        target_color_arg=target_color_arg,
        target_shape_arg=target_shape_arg,
        task_type_arg=task_type_arg,
        rng=rng,
        instruction_template=instruction_template,
    )

    if randomize_objects:
        object_specs = sample_object_specs(
            rng=rng,
            x_range=object_x_range,
            y_range=object_y_range,
            min_distance=min_object_distance,
        )
    else:
        object_specs = make_default_object_specs()

    # Push 테스트 때 target cube를 중앙/앞쪽에 두고 distractor를 옆/뒤로 보내서
    # red/blue 혼동과 중앙 정렬 문제를 줄인다.
    if task_type == "push":
        object_specs = focus_push_target_scene(object_specs, target_key)

    env = SyncSimRaccoonEnv(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
    )

    try:
        reset_multicolor_scene(
            env=env,
            object_specs=object_specs,
            target_key=target_key,
        )

        env.lockh()
        env.debug_check_current_ee_reachable()

        # Dataset collector와 동일하게 첫 observation 전에 free-joint cylinder를 안정화한다.
        if initial_settle_seconds > 0:
            env.settle_steps(seconds=initial_settle_seconds)

        write_rollout_meta(
            out_dir=out_dir,
            instruction=instruction,
            task_type=task_type,
            target_key=target_key,
            target_color=target_color,
            target_shape=target_shape,
            object_specs=object_specs,
            args={
                "xml_path": xml_path,
                "unnorm_key": unnorm_key,
                "task_type": task_type,
                "camera_name": camera_name,
                "speed": speed,
                "settle_seconds_per_action": settle_seconds_per_action,
                "initial_settle_seconds": initial_settle_seconds,
                "delta_scale": delta_scale,
                "max_delta_xyz": max_delta_xyz,
                "seed": seed,
                "object_x_range": list(object_x_range),
                "object_y_range": list(object_y_range),
                "min_object_distance": min_object_distance,
            },
        )

        print(
            f"[SCENE] instruction={instruction!r} | task={task_type!r} | target={target_key!r} | "
            f"target_xy=({object_specs[target_key]['x']:.3f}, {object_specs[target_key]['y']:.3f}) | "
            f"objects={object_specs_to_meta(object_specs)}"
        )

        obs = env.get_observation()
        step_idx = 0

        while True:
            response = request_action(
                server_url=server_url,
                instruction=instruction,
                image_rgb=obs["image"],
                unnorm_key=unnorm_key,
                timeout=request_timeout,
            )
            action = response["action"]

            # Push task 보정:
            # end-effector가 낮게 내려온 뒤에는 +y 방향 이동을 더 크게 만들어
            # 큐브가 실제로 더 많이 밀리도록 한다.
            if task_type == "push":
                action = np.asarray(action, dtype=np.float32).copy()

                if action.shape[0] >= 3:
                    ee_pose = obs.get("ee_pose", [0.0, 0.0, 1.0])
                    ee_z = float(ee_pose[2]) if len(ee_pose) >= 3 else 1.0

                    # 로봇이 충분히 낮게 내려온 뒤에만 push 보정 적용
                    if ee_z <= 0.045:
                        old_y = float(action[1])
                        min_action_y = 0.006 / max(float(delta_scale), 1e-6)
                        action[1] = max(float(action[1]) * 2.0, min_action_y)
                        print(
                            f"[PUSH BOOST] ee_z={ee_z:.3f} | "
                            f"action_y={old_y:.4f}->{float(action[1]):.4f}"
                        )

                action = action.tolist()

            try:
                exec_info = env.execute_delta_action7(
                    action=action,
                    speed=speed,
                    delta_scale=delta_scale,
                    max_delta_xyz=max_delta_xyz,
                )
                print_success_log(step_idx, exec_info)

                env.settle_steps(seconds=settle_seconds_per_action)
                obs = env.get_observation()

                frame_name = f"frame_{step_idx:06d}.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)

            except Exception as exc:
                print_fail_log(step_idx, exc)
                obs = env.get_observation()

                frame_name = f"frame_{step_idx:06d}_skipped.png"
                Image.fromarray(obs["image"]).save(out_dir / frame_name)

                step_idx += 1
                if step_idx >= max_steps:
                    print("[STOP] max_steps reached")
                    break
                continue

            step_idx += 1
            if step_idx >= max_steps:
                print("[STOP] max_steps reached")
                break

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")

    finally:
        env.close()


def clear_existing_images(out_dir: Path) -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    deleted_count = 0
    for file_path in out_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            file_path.unlink()
            deleted_count += 1

    print(f"[CLEANUP] removed {deleted_count} existing image files from {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder_cube.xml")
    parser.add_argument("--server_url", type=str, default=None, help="Direct HTTP URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="OpenVLA prompt. If omitted, generated as 'grasp the {color} {shape}'.",
    )
    parser.add_argument(
        "--target_color",
        type=str,
        default="auto",
        choices=["auto", "random", *OBJECT_COLORS],
        help="Target color. 'auto' uses the color in --instruction, or random if instruction has no color.",
    )
    parser.add_argument(
        "--target_shape",
        type=str,
        default="auto",
        choices=["auto", "random", *OBJECT_SHAPES],
        help="Target shape. 'auto' uses the shape in --instruction, or random if instruction has no shape.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="auto",
        choices=["auto", "random", *TASK_TYPES],
        help="Task type: grasp or push. 'auto' uses task in --instruction, or random if instruction has no task.",
    )
    parser.add_argument("--instruction_template", type=str, default=DEFAULT_INSTRUCTION_TEMPLATE)
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--output_dir", type=str, default="rollout_outputs")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=70)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.8)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.3)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--max_delta_xyz", type=float, default=0.005)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument(
        "--no_randomize_box",
        action="store_true",
        help="Legacy name. Disables randomization for all colored objects.",
    )
    parser.add_argument(
        "--no_randomize_objects",
        action="store_true",
        help="Disables randomization for all colored objects.",
    )

    parser.add_argument("--use_ssh_tunnel", action="store_true", help="Connect to the inference server through SSH local port forwarding")
    parser.add_argument("--ssh_host", type=str, default="qlak315.iptime.org")
    parser.add_argument("--ssh_port", type=int, default=24100)
    parser.add_argument("--ssh_user", type=str, default="root")
    parser.add_argument("--ssh_password", type=str, default=None, help="Prefer OPENVLA_SSH_PASSWORD or --ssh_ask_password")
    parser.add_argument("--ssh_ask_password", action="store_true", help="Prompt for the SSH password interactively")
    parser.add_argument("--remote_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--remote_server_port", type=int, default=8000)
    parser.add_argument("--local_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--local_server_port", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)

        if tunnel is not None:
            print(
                f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> "
                f"{args.remote_server_host}:{args.remote_server_port}"
            )

        rollout(
            xml_path=args.xml_path,
            server_url=server_url,
            instruction=args.instruction,
            unnorm_key=args.unnorm_key,
            output_dir=args.output_dir,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            use_viewer=args.use_viewer,
            camera_name=args.camera_name,
            speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            randomize_objects=not (args.no_randomize_box or args.no_randomize_objects),
            request_timeout=args.request_timeout,
            max_delta_xyz=args.max_delta_xyz,
            target_color_arg=args.target_color,
            target_shape_arg=args.target_shape,
            task_type_arg=args.task_type,
            instruction_template=args.instruction_template,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
        )


if __name__ == "__main__":
    main()
