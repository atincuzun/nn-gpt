import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ab.gpt.util.Const import conf_llm_dir, epoch_dir, nngpt_dir


def _slugify(value: str) -> str:
    cleaned = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unnamed"


def _load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override

    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_llm_conf_path(conf_name: str) -> Path:
    conf_path = Path(conf_name)
    if conf_path.is_absolute():
        return conf_path
    return conf_llm_dir / conf_path


def _prepare_variant_llm_conf(experiment_name: str, variant: dict) -> tuple[Path, str, dict]:
    source_name = variant.get("llm_conf") or variant.get("base_llm_conf")
    if not source_name:
        raise ValueError(f"Variant '{variant.get('name', 'unknown')}' must define 'llm_conf' or 'base_llm_conf'")

    base_conf_path = _resolve_llm_conf_path(source_name)
    resolved = _load_json(base_conf_path)

    llm_overrides = dict(variant.get("llm_conf_overrides", {}))
    if variant.get("moe_edit"):
        llm_overrides["moe_edit"] = _deep_merge(llm_overrides.get("moe_edit", {}), variant["moe_edit"])

    resolved = _deep_merge(resolved, llm_overrides)

    generated_dir = conf_llm_dir / "generated" / "moe_experiments" / _slugify(experiment_name)
    generated_dir.mkdir(parents=True, exist_ok=True)
    variant_name = _slugify(variant["name"])
    resolved_path = generated_dir / f"{variant_name}.json"
    with open(resolved_path, "w") as fh:
        json.dump(resolved, fh, indent=2)

    relative_conf = resolved_path.relative_to(conf_llm_dir).as_posix()
    return resolved_path, relative_conf, resolved


def _flatten_cli_args(arguments: dict) -> list[str]:
    cli_args = []
    for key, value in arguments.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if key == "target_modules":
                cli_args.extend([flag, ",".join(str(v) for v in value)])
            else:
                cli_args.append(flag)
                cli_args.extend(str(v) for v in value)
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def _extract_accuracy(eval_results):
    if isinstance(eval_results, dict):
        for key in ("accuracy", "acc"):
            value = eval_results.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    if isinstance(eval_results, (list, tuple)):
        if len(eval_results) > 1 and isinstance(eval_results[1], (int, float)):
            return float(eval_results[1])
        for value in eval_results:
            if isinstance(value, dict):
                accuracy = _extract_accuracy(value)
                if accuracy is not None:
                    return accuracy
    return None


def _summarize_epoch(epoch_path: Path) -> dict:
    synth_path = epoch_path / "synth_nn"
    moe_meta_path = epoch_path / "moe" / "moe_edit_metadata.json"
    model_summaries = []
    accuracies = []

    total_generated = 0
    if synth_path.exists():
        for model_dir in sorted(p for p in synth_path.iterdir() if p.is_dir()):
            total_generated += 1
            eval_info_path = model_dir / "eval_info.json"
            error_path = model_dir / "error.txt"
            accuracy = None
            status = "pending"
            if eval_info_path.exists():
                try:
                    eval_info = _load_json(eval_info_path)
                    accuracy = _extract_accuracy(eval_info.get("eval_results"))
                    status = "success" if accuracy is not None else "evaluated_without_accuracy"
                except Exception as exc:
                    status = f"eval_info_error: {exc}"
            elif error_path.exists():
                status = "failed"

            if accuracy is not None:
                accuracies.append(accuracy)

            model_summaries.append({
                "model_id": model_dir.name,
                "accuracy": accuracy,
                "status": status,
            })

    moe_metadata = _load_json(moe_meta_path) if moe_meta_path.exists() else None
    successful = len(accuracies)
    return {
        "epoch": epoch_path.name,
        "path": str(epoch_path),
        "total_generated": total_generated,
        "successful": successful,
        "success_rate": successful / total_generated if total_generated else 0.0,
        "best_accuracy": max(accuracies) if accuracies else None,
        "avg_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else None,
        "accuracies": accuracies,
        "models": model_summaries,
        "moe_metadata": moe_metadata,
    }


def _summarize_variant_artifacts(variant_dir: Path, resolved_conf: dict, variant: dict) -> dict:
    archived_epoch_root = variant_dir / "epoch_artifacts"
    epoch_summaries = []
    all_accuracies = []
    if archived_epoch_root.exists():
        for epoch_path in sorted(p for p in archived_epoch_root.iterdir() if p.is_dir()):
            epoch_summary = _summarize_epoch(epoch_path)
            epoch_summaries.append(epoch_summary)
            all_accuracies.extend(epoch_summary["accuracies"])

    total_generated = sum(item["total_generated"] for item in epoch_summaries)
    successful = sum(item["successful"] for item in epoch_summaries)
    cycle_results_path = variant_dir / "cycle_results.json"
    cycle_results = _load_json(cycle_results_path) if cycle_results_path.exists() else None

    aggregate = {
        "epochs_evaluated": len(epoch_summaries),
        "total_generated": total_generated,
        "successful": successful,
        "success_rate": successful / total_generated if total_generated else 0.0,
        "best_accuracy": max(all_accuracies) if all_accuracies else None,
        "avg_accuracy": (sum(all_accuracies) / len(all_accuracies)) if all_accuracies else None,
    }

    return {
        "variant": variant,
        "resolved_llm_conf": resolved_conf,
        "cycle_results": cycle_results,
        "epoch_summaries": epoch_summaries,
        "aggregate": aggregate,
    }


def _archive_run_outputs(variant_dir: Path) -> None:
    variant_dir.mkdir(parents=True, exist_ok=True)
    source_epoch_dir = epoch_dir()
    archived_epoch_dir = variant_dir / "epoch_artifacts"
    if archived_epoch_dir.exists():
        shutil.rmtree(archived_epoch_dir)
    if source_epoch_dir.exists():
        shutil.copytree(source_epoch_dir, archived_epoch_dir)

    cycle_results_path = nngpt_dir / "cycle_results.json"
    if cycle_results_path.exists():
        shutil.copy2(cycle_results_path, variant_dir / "cycle_results.json")


def _tee_subprocess(command: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def _ranking_key(summary: dict, ranking: dict) -> tuple:
    aggregate = summary.get("aggregate", {})
    metrics = [
        ranking.get("primary", "best_accuracy"),
        ranking.get("secondary", "avg_accuracy"),
        ranking.get("tertiary", "success_rate"),
    ]
    values = []
    for metric in metrics:
        value = aggregate.get(metric)
        values.append(float("-inf") if value is None else float(value))
    return tuple(values)


def _build_leaderboard_markdown(experiment_name: str, summaries: list[dict]) -> str:
    lines = [
        f"# {experiment_name}",
        "",
        "| Rank | Variant | Best Accuracy | Avg Accuracy | Success Rate | Generated | Successful |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, summary in enumerate(summaries, start=1):
        aggregate = summary.get("aggregate", {})
        variant_name = summary.get("variant", {}).get("name", f"variant_{idx}")
        best_accuracy = aggregate.get("best_accuracy")
        avg_accuracy = aggregate.get("avg_accuracy")
        success_rate = aggregate.get("success_rate")
        total_generated = aggregate.get("total_generated", 0)
        successful = aggregate.get("successful", 0)
        lines.append(
            "| {rank} | {variant} | {best} | {avg} | {rate} | {total} | {successful} |".format(
                rank=idx,
                variant=variant_name,
                best="-" if best_accuracy is None else f"{best_accuracy:.4f}",
                avg="-" if avg_accuracy is None else f"{avg_accuracy:.4f}",
                rate=f"{success_rate:.4f}",
                total=total_generated,
                successful=successful,
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Run standalone NNGPT MoE variant experiments.")
    parser.add_argument(
        "--experiment_conf",
        type=str,
        default="moe_variant_sweep_example.json",
        help="Experiment config JSON relative to conf/llm or an absolute path.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional override for the experiment output directory name.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved commands without executing the generate/evaluate/finetune loop.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop the sweep immediately if one MoE variant fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_conf_path = _resolve_llm_conf_path(args.experiment_conf)
    experiment_conf = _load_json(experiment_conf_path)

    experiment_name = args.experiment_name or experiment_conf.get("experiment_name") or experiment_conf_path.stem
    experiment_slug = _slugify(experiment_name)
    experiment_dir = nngpt_dir / "moe_experiments" / experiment_slug
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shared_tune_args = experiment_conf.get("shared_tune_args", {})
    ranking = experiment_conf.get("ranking", {})
    variants = experiment_conf.get("variants", [])
    if not variants:
        raise ValueError(f"No variants defined in {experiment_conf_path}")

    project_root = Path(__file__).resolve().parents[2]
    summaries = []
    for index, variant in enumerate(variants, start=1):
        variant_name = variant.get("name") or f"variant_{index}"
        variant_slug = _slugify(variant_name)
        variant_dir = experiment_dir / variant_slug
        resolved_path, resolved_rel_conf, resolved_conf = _prepare_variant_llm_conf(experiment_slug, variant)

        variant_tune_args = _deep_merge(shared_tune_args, variant.get("tune_args", {}))
        command = [
            sys.executable,
            "-m",
            "ab.gpt.TuneNNGen",
            "--llm_conf",
            resolved_rel_conf,
            *_flatten_cli_args(variant_tune_args),
        ]

        manifest = {
            "experiment_name": experiment_name,
            "variant_name": variant_name,
            "variant_index": index,
            "command": command,
            "resolved_llm_conf": resolved_rel_conf,
            "resolved_llm_conf_path": str(resolved_path),
            "shared_tune_args": shared_tune_args,
            "variant_tune_args": variant.get("tune_args", {}),
            "variant": variant,
        }
        variant_dir.mkdir(parents=True, exist_ok=True)
        with open(variant_dir / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)

        print(f"\n[MoE Sweep] ({index}/{len(variants)}) Variant: {variant_name}")
        print(f"[MoE Sweep] Resolved llm_conf: {resolved_rel_conf}")
        print(f"[MoE Sweep] Command: {' '.join(command)}")

        return_code = 0
        if not args.dry_run:
            return_code = _tee_subprocess(command, variant_dir / "run.log", cwd=project_root)
            _archive_run_outputs(variant_dir)

        summary = _summarize_variant_artifacts(variant_dir, resolved_conf, variant)
        summary["return_code"] = return_code
        summary["command"] = command
        summary["resolved_llm_conf"] = resolved_conf
        with open(variant_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        summaries.append(summary)

        if return_code != 0 and args.stop_on_error:
            print(f"[MoE Sweep] Stopping after failure in variant '{variant_name}'")
            break

    ranked_summaries = sorted(summaries, key=lambda item: _ranking_key(item, ranking), reverse=True)
    leaderboard = {
        "experiment_name": experiment_name,
        "experiment_conf": str(experiment_conf_path),
        "ranking": ranking,
        "variants": ranked_summaries,
    }
    with open(experiment_dir / "leaderboard.json", "w") as fh:
        json.dump(leaderboard, fh, indent=2)
    with open(experiment_dir / "leaderboard.md", "w") as fh:
        fh.write(_build_leaderboard_markdown(experiment_name, ranked_summaries))

    if ranked_summaries:
        winner = ranked_summaries[0]
        print("\n[MoE Sweep] Best variant:")
        print(json.dumps({
            "variant": winner.get("variant", {}).get("name"),
            "aggregate": winner.get("aggregate", {}),
            "return_code": winner.get("return_code"),
        }, indent=2))


if __name__ == "__main__":
    main()
