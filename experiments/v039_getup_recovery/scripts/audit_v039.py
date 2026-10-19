"""Audit the local v039 runtime, official evidence, and 23-DoF model contract."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v039_getup_recovery.artifacts import write_json  # noqa: E402
from experiments.v039_getup_recovery.getup_env import T1GetUpEnv  # noqa: E402


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch",
            "mujoco",
            "gymnasium",
            "stable_baselines3",
            "isaacgym",
            "isaaclab",
            "rsl_rl",
        )
    }
    env = T1GetUpEnv(stage=0)
    observation, reset_info = env.reset(seed=39000)
    model_info = {
        "nq": int(env.model.nq),
        "nv": int(env.model.nv),
        "nu": int(env.model.nu),
        "observation_dim": int(observation.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
        "reset_info": reset_info,
    }
    env.close()
    torch_info: dict[str, object] = {}
    if modules["torch"]:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            if torch.cuda.is_available()
            else None,
        }
    mem_total_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        first = meminfo.read_text(encoding="utf-8").splitlines()[0].split()
        mem_total_kib = int(first[1])
    payload = {
        "experiment": "v039_getup_recovery",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "glibc": _command_output(["ldd", "--version"]),
        "wsl_memory_gb": round(mem_total_kib / 1024**2, 2) if mem_total_kib else None,
        "modules": modules,
        "torch": torch_info,
        "model": model_info,
        "official_sources": {
            "booster_train": {
                "commit": "651b7a53f2ffaf2d5629d0604d065cc385e29c6b",
                "finding": "BeyondMimic robot task directories contain K1 only; no T1 get-up task.",
                "url": "https://github.com/BoosterRobotics/booster_train/tree/651b7a53f2ffaf2d5629d0604d065cc385e29c6b",
            },
            "booster_assets": {
                "commit": "508cbee6ca9ae6fbc8c0b38dd58785a6f3fc61a2",
                "finding": "T1 models exist, but all published motion CSV files are K1 motions.",
                "url": "https://github.com/BoosterRobotics/booster_assets/tree/508cbee6ca9ae6fbc8c0b38dd58785a6f3fc61a2",
            },
            "booster_deploy": {
                "commit": "563e7c37e7468b3b96ad9f84f9e61becea3b7fb8",
                "finding": "The only published T1 policy artifact is t1_walk.pt; no T1 get-up policy.",
                "url": "https://github.com/BoosterRobotics/booster_deploy/tree/563e7c37e7468b3b96ad9f84f9e61becea3b7fb8",
            },
            "booster_gym": {
                "commit": "da396a06d6eed99e2de72d7749c48ee8748950f9",
                "finding": "The T1 task is locomotion-only, imports Isaac Gym, and terminates when base height is below the fall threshold; no get-up task is published.",
                "url": "https://github.com/BoosterRobotics/booster_gym/tree/da396a06d6eed99e2de72d7749c48ee8748950f9",
            },
            "robocup_demo": {
                "commit": "fa4f31ccb4428a9e60e90b4f54eb80499caeb0ca",
                "finding": "CheckAndStandUp calls the closed firmware API LocoApiId::kGetUp; no trajectory or weights are published.",
                "url": "https://github.com/BoosterRobotics/robocup_demo/blob/fa4f31ccb4428a9e60e90b4f54eb80499caeb0ca/src/brain/src/robot_client.cpp",
            },
        },
        "isaac_lab_hardware_gate": {
            "official_minimum_ram_gb": 32,
            "official_minimum_vram_gb": 16,
            "local_below_minimum": bool(
                (mem_total_kib is not None and mem_total_kib / 1024**2 < 32)
                or (torch_info.get("cuda_memory_gb") or 0) < 16
            ),
            "source": "https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html",
        },
    }
    output = write_json(
        Path(__file__).resolve().parents[1] / "metrics" / "v039_runtime_audit.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
