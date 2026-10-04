#!/usr/bin/env python3
"""Run a documented, staged SpatialRXNN experiment from a JSON config."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment JSON file.")
    parser.add_argument("--stage", action="append", dest="stages", help="Run only this stage (repeatable).")
    parser.add_argument("--epochs", type=int, help="Override epochs for every selected stage.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without importing Chemprop.")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose model checkpoint already exists.")
    return parser


def load_config(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("version") != 1:
        raise ValueError("Experiment config must declare version 1")
    if not isinstance(config.get("stages"), list) or not config["stages"]:
        raise ValueError("Experiment config must contain at least one stage")
    return config


def project_root(config_path: Path, config: Mapping[str, object]) -> Path:
    configured = Path(str(config.get("project_root", ".")))
    if not configured.is_absolute():
        configured = config_path.resolve().parent / configured
    return configured.resolve()


def cli_arguments(arguments: Mapping[str, object]) -> List[str]:
    result: List[str] = []
    for name, value in arguments.items():
        flag = f"--{name}"
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result.append(flag)
            if isinstance(value, list):
                result.extend(str(item) for item in value)
            else:
                result.append(str(value))
    return result


def checkpoint_path(save_dir: Path) -> Path:
    return save_dir / "fold_0" / "model_0" / "model.pt"


def prepare_stages(
    config_path: Path,
    config: Mapping[str, object],
    selected: Optional[Sequence[str]] = None,
    epochs: Optional[int] = None,
) -> List[Dict[str, object]]:
    root = project_root(config_path, config)
    common = dict(config.get("common", {}))
    raw_stages = config["stages"]
    names = [stage.get("name") for stage in raw_stages]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Every stage must have a unique, non-empty name")
    unknown = set(selected or ()) - set(names)
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(sorted(unknown))}")

    prepared: List[Dict[str, object]] = []
    outputs: Dict[str, Path] = {}
    for raw_stage in raw_stages:
        name = raw_stage["name"]
        arguments = dict(common)
        arguments.update(raw_stage.get("arguments", {}))
        for path_key in ("data_path", "spatial_features_path", "edge_types_path"):
            if path_key in raw_stage:
                arguments[path_key] = raw_stage[path_key]
        if epochs is not None:
            arguments["epochs"] = epochs

        for path_key in ("data_path", "spatial_features_path", "edge_types_path"):
            if path_key in arguments:
                value = Path(str(arguments[path_key]))
                arguments[path_key] = str(value if value.is_absolute() else root / value)

        save_dir = Path(raw_stage["save_dir"])
        save_dir = save_dir if save_dir.is_absolute() else root / save_dir
        arguments["save_dir"] = str(save_dir)

        dependency = raw_stage.get("checkpoint_from")
        if dependency:
            if dependency not in outputs:
                raise ValueError(f"Stage {name!r} depends on unknown or later stage {dependency!r}")
            arguments["checkpoint_path"] = str(outputs[dependency])

        output = checkpoint_path(save_dir)
        outputs[name] = output
        prepared.append({"name": name, "arguments": arguments, "output": output})

    if selected:
        selected_set = set(selected)
        prepared = [stage for stage in prepared if stage["name"] in selected_set]
    return prepared


def printable_command(arguments: Mapping[str, object]) -> str:
    return shlex.join(["chemprop_train", *cli_arguments(arguments)])


def validate_inputs(arguments: Mapping[str, object]) -> None:
    data_path = Path(str(arguments["data_path"]))
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    checkpoint = arguments.get("checkpoint_path")
    if checkpoint and not Path(str(checkpoint)).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    stages = prepare_stages(args.config, config, args.stages, args.epochs)

    if args.dry_run:
        for stage in stages:
            print(f"[{stage['name']}]\n{printable_command(stage['arguments'])}\n")
        return 0

    try:
        from chemprop.args import TrainArgs
        from chemprop.train.cross_validate import cross_validate
        from chemprop.train.run_training import run_training
    except ImportError as error:
        raise SystemExit("Install the SpatialRXNN environment before running experiments.") from error

    results = {}
    for stage in stages:
        name = str(stage["name"])
        output = Path(stage["output"])
        if args.resume and output.exists():
            print(f"[{name}] checkpoint exists; skipping")
            continue
        arguments = stage["arguments"]
        validate_inputs(arguments)
        print(f"[{name}] starting")
        train_args = TrainArgs().parse_args(cli_arguments(arguments))
        mean_score, std_score = cross_validate(args=train_args, train_func=run_training)
        results[name] = {"mean_score": float(mean_score), "std_score": float(std_score)}

    if results:
        summary_path = project_root(args.config, config) / "outputs" / "pipeline_results.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
