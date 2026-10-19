"""Train the exact-RCSS T1 getup task, optionally bootstrapped from v044."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.v045_rcss_t1_getup_adaptation.rcss_mjlab_task import (  # noqa: E402
    TASK_ID,
    register_v045_task,
)
from experiments.v045_rcss_t1_getup_adaptation.stability_watchdog import (  # noqa: E402
    install_v045_watchdog,
)
from experiments.v045_rcss_t1_getup_adaptation.training_profiles import (  # noqa: E402
    BASE_COM_PROFILES,
    RESET_PROFILES,
    SOLVER_PROFILES,
    apply_training_profiles,
)


PROFILE_SELECTION_SCHEMA = "v045_profile_selection_proof_v1"
FORMAL_ENVIRONMENTS = 4096
FORMAL_ITERATIONS = 3000
FORMAL_SEED = 45000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=45000)
    parser.add_argument("--run-name", default="v045_rcss_adaptation_full_r1")
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--bootstrap-selection-proof", type=Path)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nan-guard", action="store_true")
    parser.add_argument(
        "--engineering-bootstrap-std",
        type=float,
        help="smoke-only reset of the warm-start Gaussian exploration std",
    )
    parser.add_argument(
        "--engineering-entropy-coef",
        type=float,
        help="smoke-only PPO entropy coefficient override",
    )
    parser.add_argument(
        "--engineering-learning-rate",
        type=float,
        help="smoke-only PPO learning-rate override for conservative fine-tuning",
    )
    parser.add_argument(
        "--solver-profile", choices=tuple(SOLVER_PROFILES), default="rcss_mjb"
    )
    parser.add_argument(
        "--base-com-profile", choices=tuple(BASE_COM_PROFILES), default="off"
    )
    parser.add_argument(
        "--reset-profile", choices=tuple(RESET_PROFILES), default="rcss_eval_v2"
    )
    parser.add_argument("--profile-selection-proof", type=Path)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.iterations <= 0:
        parser.error("num-envs and iterations must be positive")
    if args.bootstrap_selection_proof is not None and args.bootstrap is None:
        parser.error("--bootstrap-selection-proof requires --bootstrap")
    if args.engineering_bootstrap_std is not None:
        if not args.engineering_smoke or args.bootstrap is None:
            parser.error(
                "--engineering-bootstrap-std requires an engineering bootstrap"
            )
        if not 0.05 <= args.engineering_bootstrap_std <= 1.0:
            parser.error("engineering bootstrap std must be in [0.05, 1.0]")
    if args.engineering_entropy_coef is not None:
        if not args.engineering_smoke:
            parser.error("entropy override is allowed only for engineering runs")
        if not 0.0 <= args.engineering_entropy_coef <= 0.01:
            parser.error("engineering entropy coefficient must be in [0, 0.01]")
    if args.engineering_learning_rate is not None:
        if not args.engineering_smoke:
            parser.error("learning-rate override is allowed only for engineering runs")
        if not 1.0e-6 <= args.engineering_learning_rate <= 1.0e-3:
            parser.error("engineering learning rate must be in [1e-6, 1e-3]")
    if (
        args.bootstrap is not None
        and args.bootstrap_selection_proof is None
        and not args.engineering_smoke
    ):
        parser.error(
            "formal bootstrap requires --bootstrap-selection-proof; use "
            "--engineering-smoke only for a non-promotable probe"
        )
    formal_shape = (
        args.num_envs == FORMAL_ENVIRONMENTS
        and args.iterations == FORMAL_ITERATIONS
        and args.seed == FORMAL_SEED
        and not args.engineering_smoke
    )
    if not args.engineering_smoke and not formal_shape:
        parser.error(
            "non-formal dimensions/seeds require --engineering-smoke; only "
            "4096 environments x 3000 iterations with seed 45000 is formal"
        )
    if formal_shape and args.profile_selection_proof is None:
        parser.error(
            "formal v045 training is fail-closed until an immutable controlled-A/B "
            "profile selection proof is provided"
        )
    return args


def _validate_selection_proof(checkpoint: Path, proof_path: Path) -> dict[str, str]:
    checkpoint = checkpoint.resolve()
    proof_path = proof_path.resolve()
    if not proof_path.is_file():
        raise FileNotFoundError(proof_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != "v044_source_candidate_eval_v2":
        raise ValueError("bootstrap selection proof schema mismatch")
    if not proof.get("formal_source_shortlist_protocol", False):
        raise ValueError("bootstrap selection proof is not the formal shortlist")
    if proof.get("seeds") != [44700, 44701, 44702]:
        raise ValueError("bootstrap selection proof has the wrong seed set")
    if proof.get("num_environments") != 256 or proof.get("steps") != 299:
        raise ValueError("bootstrap selection proof has the wrong evaluation size")
    selected = proof.get("selected_checkpoint_for_target_adaptation")
    if not isinstance(selected, dict):
        raise ValueError("bootstrap selection proof has no selected checkpoint")
    selected_path = selected.get("path")
    if not isinstance(selected_path, str) or not selected_path:
        raise ValueError("bootstrap selection proof has an invalid checkpoint path")
    if Path(selected_path).resolve() != checkpoint:
        raise ValueError("bootstrap checkpoint does not match the selected candidate")
    checkpoint_sha256 = _sha256(checkpoint)
    if selected.get("sha256") != checkpoint_sha256:
        raise ValueError("bootstrap checkpoint hash does not match selection proof")
    return {
        "path": str(proof_path),
        "sha256": _sha256(proof_path),
        "selected_checkpoint_sha256": checkpoint_sha256,
    }


def _prepare_bootstrap(
    checkpoint: Path,
    log_root: Path,
    *,
    initial_learning_rate: float,
    selection_proof: dict[str, str] | None,
    bootstrap_std: float | None,
) -> tuple[str, str]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checksum = _sha256(checkpoint)
    transform = (
        "network_warm_start_fresh_state_fixed_exploration_v3"
        if bootstrap_std is not None
        else "network_warm_start_with_fresh_training_state_v2"
    )
    proof_tag = (
        selection_proof["sha256"][:12] if selection_proof is not None else "unproven"
    )
    std_tag = (
        f"_std{int(round(bootstrap_std * 1000.0)):04d}"
        if bootstrap_std is not None
        else ""
    )
    lr_tag = (
        f"_lr{int(round(initial_learning_rate * 1.0e9)):07d}"
        if bootstrap_std is not None
        else ""
    )
    version_tag = "v3" if bootstrap_std is not None else "v2"
    run_name = (
        f"bootstrap_{checksum[:16]}_{proof_tag}{std_tag}{lr_tag}_warmstart_{version_tag}"
    )
    run_dir = log_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{checkpoint.stem}.warmstart{checkpoint.suffix}"
    target = run_dir / target_name

    import torch

    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    missing = sorted(required.difference(source_payload))
    if missing:
        raise ValueError(f"bootstrap checkpoint is missing fields: {missing}")
    source_iteration = int(source_payload["iter"])

    if not target.exists():
        # Domain adaptation must inherit the learned actor/critic, action
        # distribution and observation normalizer, but not the source-domain
        # optimizer momentum, learning-rate decay, iteration number or
        # curriculum counter.  A full "resume" was tested and incorrectly
        # jumped v045 straight to the late energy/smoothness curriculum.
        optimizer = source_payload["optimizer_state_dict"]
        optimizer["state"] = {}
        for group in optimizer["param_groups"]:
            group["lr"] = float(initial_learning_rate)
        source_payload["iter"] = 0
        if bootstrap_std is not None:
            log_std = source_payload["actor_state_dict"].get(
                "distribution.log_std_param"
            )
            if log_std is None or tuple(log_std.shape) != (23,):
                raise ValueError("bootstrap actor has no canonical 23D log std")
            log_std.fill_(math.log(float(bootstrap_std)))
        source_payload["infos"] = {
            "env_state": {"common_step_counter": 0},
            "warm_start_provenance": {
                "source": str(checkpoint),
                "source_sha256": checksum,
                "source_iteration": source_iteration,
                "optimizer_state_reset": True,
                "curriculum_counter_reset": True,
                "learning_rate_reset_to": float(initial_learning_rate),
                "selection_proof": selection_proof,
                "bootstrap_std": bootstrap_std,
            },
        }
        torch.save(source_payload, target)
    cached_checksum = _sha256(target)
    proof = run_dir / "bootstrap_provenance.json"
    payload = {
        "source": str(checkpoint),
        "source_sha256": checksum,
        "source_iteration": source_iteration,
        "cached_checkpoint": str(target),
        "cached_sha256": cached_checksum,
        "transform": transform,
        "optimizer_state_reset": True,
        "curriculum_counter_reset": True,
        "target_iteration": 0,
        "learning_rate_reset_to": float(initial_learning_rate),
        "selection_proof": selection_proof,
        "bootstrap_std": bootstrap_std,
    }
    if proof.exists():
        if json.loads(proof.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"existing bootstrap proof mismatch: {proof}")
    else:
        proof.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return run_name, target.name


def _reference(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _validate_profile_selection(
    proof_path: Path,
    *,
    solver_profile: str,
    base_com_profile: str,
    reset_profile: str,
) -> dict[str, str]:
    """Require a sealed controlled-A/B result before any formal v045 run."""

    proof_path = proof_path.expanduser().resolve()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("schema_version") != PROFILE_SELECTION_SCHEMA:
        raise ValueError("v045 profile-selection proof schema mismatch")
    required_true = (
        "controlled_ab_completed",
        "rcss_transfer_completed",
        "formal_profile_selection_passed",
    )
    if not all(proof.get(name) is True for name in required_true):
        raise ValueError("v045 profile-selection proof is incomplete or failed")
    selected = proof.get("selected_profiles")
    expected = {
        "solver": solver_profile,
        "base_com": base_com_profile,
        "reset": reset_profile,
    }
    if selected != expected:
        raise ValueError("requested profiles differ from the sealed A/B selection")
    if selected.get("solver") != "rcss_mjb" or selected.get("reset") != "rcss_eval_v2":
        raise ValueError("formal v045 requires the exact RCSS solver and reset domain")
    sources = proof.get("evidence")
    if not isinstance(sources, list) or not sources:
        raise ValueError("profile-selection proof has no controlled evidence")
    for index, reference in enumerate(sources):
        if not isinstance(reference, dict):
            raise ValueError(f"profile-selection evidence {index} is invalid")
        path = Path(reference.get("path", "")).expanduser().resolve()
        if not path.is_file() or reference.get("sha256") != _sha256(path):
            raise ValueError(f"profile-selection evidence changed: {path}")
    return _reference(proof_path)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _run_training_with_watchdog(
    cfg: Any,
    log_dir: Path,
    *,
    training_profile: dict[str, Any],
    run_contract: dict[str, Any],
) -> None:
    """Run one local worker and install the guard before the first policy step."""

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import dump_yaml, get_checkpoint_path
    from mjlab.utils.torch import configure_torch_backends

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else "cuda:0"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    configure_torch_backends()
    cfg.env.seed = cfg.agent.seed
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True

    log_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(log_dir / "training_profile.json", training_profile)
    _write_json_exclusive(log_dir / "run_contract.json", run_contract)
    raw_env = None
    wrapped_env = None
    watchdog = None
    failure: BaseException | None = None
    try:
        raw_env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=None)
        watchdog = install_v045_watchdog(
            raw_env, log_dir, training_profile=training_profile
        )

        resume_path = None
        if cfg.agent.resume:
            resume_path = get_checkpoint_path(
                log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint
            )

        agent_cfg = asdict(cfg.agent)
        env_cfg = asdict(cfg.env)
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
        wrapped_env = RslRlVecEnvWrapper(
            raw_env, clip_actions=cfg.agent.clip_actions
        )
        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped_env, agent_cfg, str(log_dir), device)
        runner.add_git_repo_to_log(__file__)
        if resume_path is not None:
            print(f"[INFO] Loading warm-start checkpoint: {resume_path}")
            runner.load(str(resume_path))
        runner.learn(
            num_learning_iterations=cfg.agent.max_iterations,
            init_at_random_ep_len=True,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    except BaseException as exc:
        failure = exc
        raise
    finally:
        finalize_error = None
        if watchdog is not None:
            try:
                watchdog.finalize(
                    status="completed" if failure is None else "failed",
                    exception=failure,
                )
            except BaseException as exc:
                finalize_error = exc
        if wrapped_env is not None:
            wrapped_env.close()
        elif raw_env is not None:
            raw_env.close()
        if failure is None and finalize_error is not None:
            raise finalize_error


def _launch_training_with_watchdog(
    cfg: Any,
    *,
    cpu: bool,
    training_profile: dict[str, Any],
    run_contract: dict[str, Any],
) -> Path:
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if cpu else "0"
    os.environ["MUJOCO_GL"] = "egl"
    log_root = Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.agent.run_name:
        name += f"_{cfg.agent.run_name}"
    log_dir = log_root / name
    _run_training_with_watchdog(
        cfg,
        log_dir,
        training_profile=training_profile,
        run_contract=run_contract,
    )
    return log_dir


def main() -> int:
    args = _parse_args()
    register_v045_task()
    from mjlab.scripts.train import TrainConfig

    cfg = TrainConfig.from_task(TASK_ID)
    cfg.env.scene.num_envs = args.num_envs
    cfg.env.scene.terrain.num_envs = args.num_envs
    cfg.env.seed = args.seed
    cfg.agent.seed = args.seed
    cfg.agent.max_iterations = args.iterations
    cfg.agent.run_name = args.run_name
    if args.engineering_entropy_coef is not None:
        cfg.agent.algorithm.entropy_coef = float(args.engineering_entropy_coef)
    if args.engineering_learning_rate is not None:
        cfg.agent.algorithm.learning_rate = float(args.engineering_learning_rate)
    # Training evidence must remain local and reproducible; do not require a
    # W&B account or network access for any smoke or formal run.
    cfg.agent.logger = "tensorboard"
    cfg = replace(
        cfg,
        gpu_ids=None if args.cpu else [0],
        enable_nan_guard=(args.nan_guard or not args.engineering_smoke),
    )
    training_profile = apply_training_profiles(
        cfg.env,
        solver_profile=args.solver_profile,
        base_com_profile=args.base_com_profile,
        reset_profile=args.reset_profile,
    )
    profile_selection = (
        _validate_profile_selection(
            args.profile_selection_proof,
            solver_profile=args.solver_profile,
            base_com_profile=args.base_com_profile,
            reset_profile=args.reset_profile,
        )
        if args.profile_selection_proof is not None
        else None
    )

    if args.bootstrap is not None:
        selection_proof = (
            _validate_selection_proof(args.bootstrap, args.bootstrap_selection_proof)
            if args.bootstrap_selection_proof is not None
            else None
        )
        log_root = Path.cwd() / "logs" / "rsl_rl" / cfg.agent.experiment_name
        load_run, load_checkpoint = _prepare_bootstrap(
            args.bootstrap,
            log_root,
            initial_learning_rate=cfg.agent.algorithm.learning_rate,
            selection_proof=selection_proof,
            bootstrap_std=args.engineering_bootstrap_std,
        )
        cfg.agent.resume = True
        cfg.agent.load_run = f"^{load_run}$"
        cfg.agent.load_checkpoint = f"^{load_checkpoint}$"

    formal_training = bool(
        args.num_envs == FORMAL_ENVIRONMENTS
        and args.iterations == FORMAL_ITERATIONS
        and args.seed == FORMAL_SEED
        and not args.engineering_smoke
    )
    if formal_training and profile_selection is None:
        raise RuntimeError("formal v045 training lacks a profile-selection proof")
    run_contract = {
        "schema_version": "v045_training_run_contract_v1",
        "formal_training_run": formal_training,
        "engineering_smoke": bool(args.engineering_smoke),
        "num_environments": int(args.num_envs),
        "iterations": int(args.iterations),
        "environment_seed": int(args.seed),
        "agent_seed": int(args.seed),
        "nan_guard_enabled": bool(cfg.enable_nan_guard),
        "engineering_agent_overrides": {
            "bootstrap_std": args.engineering_bootstrap_std,
            "entropy_coef": args.engineering_entropy_coef,
            "learning_rate": args.engineering_learning_rate,
        },
        "training_profile": training_profile,
        "profile_selection_proof": profile_selection,
        "entrypoint": _reference(Path(__file__)),
    }
    log_dir = _launch_training_with_watchdog(
        cfg,
        cpu=args.cpu,
        training_profile=training_profile,
        run_contract=run_contract,
    )
    print(json.dumps({"run_directory": str(log_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
