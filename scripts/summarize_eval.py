import argparse
import csv
from collections import defaultdict
from pathlib import Path

SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def parse_results_csv(csv_path: Path) -> dict:
    results = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"]
            success = int(row["success"])
            results[task].append(success)
    return results


def compute_suite_stats(results: dict) -> tuple[float, int, int]:
    total_success = 0
    total_trials = 0
    for outcomes in results.values():
        total_success += sum(outcomes)
        total_trials += len(outcomes)
    success_rate = total_success / total_trials if total_trials > 0 else 0.0
    return success_rate, total_success, total_trials


def discover_evals(eval_logs_path: Path) -> dict:
    evals = defaultdict(lambda: defaultdict(dict))
    for model_dir in sorted(eval_logs_path.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for ckpt_dir in sorted(model_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            ckpt_name = ckpt_dir.name
            for suite_dir in sorted(ckpt_dir.iterdir()):
                if not suite_dir.is_dir():
                    continue
                suite_name = suite_dir.name
                results_csv = suite_dir / "results.csv"
                if results_csv.exists():
                    evals[model_name][ckpt_name][suite_name] = results_csv
    return evals


def get_ordered_suites(available_suites: set) -> list:
    ordered = [s for s in SUITE_ORDER if s in available_suites]
    remaining = sorted(available_suites - set(SUITE_ORDER))
    return ordered + remaining


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_logs", type=Path, default=Path("./eval_logs"))
    args = parser.parse_args()

    evals = discover_evals(args.eval_logs)

    if not evals:
        print("No evaluations found.")
        return

    for model_name in sorted(evals.keys()):
        checkpoints = evals[model_name]
        all_suites = set()
        for ckpt_suites in checkpoints.values():
            all_suites.update(ckpt_suites.keys())
        suites = get_ordered_suites(all_suites)

        print(f"\nModel: {model_name}")
        col_width = 18
        header = f"{'Checkpoint':<20}" + "".join(f"{s:>{col_width}}" for s in suites) + f"{'Average':>{col_width}}"
        print(header)
        print("-" * len(header))

        for ckpt_name in sorted(
            checkpoints.keys(),
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
        ):
            row = f"{ckpt_name:<20}"
            rates = []
            for suite in suites:
                if suite in checkpoints[ckpt_name]:
                    csv_path = checkpoints[ckpt_name][suite]
                    results = parse_results_csv(csv_path)
                    rate, succ, total = compute_suite_stats(results)
                    rates.append(rate)
                    cell = f"{rate * 100:.1f}% ({succ}/{total})"
                else:
                    cell = "-"
                row += f"{cell:>{col_width}}"
            avg = sum(rates) / len(rates) if rates else 0.0
            row += f"{avg * 100:.1f}%".rjust(col_width)
            print(row)


if __name__ == "__main__":
    main()
