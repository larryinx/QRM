import argparse
import glob
import json
import os
from os.path import basename, dirname, join

from prettytable import HRuleStyle, PrettyTable

from qrm import PROJECT_ROOT


def list_to_table(data_list, headers=None):
    table = PrettyTable()
    table.hrules = HRuleStyle.ALL

    max_list_length = max(len(value) for value in data_list)

    if headers is None:
        headers = [f"value{i + 1}" for i in range(max_list_length)]

    table.field_names = headers

    for value_list in data_list:
        row = value_list + [""] * (max_list_length - len(value_list))
        table.add_row(row)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    exp_folder_list = os.listdir(join(PROJECT_ROOT, "results"))
    exp_folder_list.sort()

    print_score_list = []
    for exp_folder in exp_folder_list:
        exp_folder_path = join(PROJECT_ROOT, "results", exp_folder)
        if not os.path.isdir(exp_folder_path):
            continue

        glob_pattern = join(exp_folder_path, "checkpoint-*/test_metrics.json")
        if args.compile:
            glob_pattern = glob_pattern.replace(".json", "_compile.json")
        test_metrics_list = glob.glob(glob_pattern)

        if len(test_metrics_list) == 0:
            glob_pattern = join(exp_folder_path, "checkpoint-*/trainer_state.json")
            test_metrics_list = glob.glob(glob_pattern)

        if len(test_metrics_list) == 0:
            continue

        test_metrics_list.sort(key=lambda x: int(basename(dirname(x)).split("-")[-1]))

        for test_metrics_path in test_metrics_list:
            with open(test_metrics_path, "r") as f:
                test_metrics = json.load(f)
            if test_metrics_path.endswith("trainer_state.json"):
                test_metrics = test_metrics["log_history"][-1]

            ckpt_name = basename(dirname(test_metrics_path))
            exp_name = basename(dirname(dirname(test_metrics_path)))

            try:
                print_score_list.append(
                    [
                        exp_name,
                        ckpt_name,
                        f"{test_metrics['eval_accuracy']:.2%}",
                        f"{test_metrics['eval_exact_accuracy']:.2%}",
                        f"{test_metrics['eval_lm_loss']:.6f}",
                        f"{test_metrics['eval_q_halt_accuracy']:.2%}",
                        f"{test_metrics['eval_q_halt_loss']:.6f}",
                        test_metrics["eval_steps"],
                        f"{test_metrics['eval_runtime']:.1f}",
                        f"{test_metrics['eval_samples_per_second']:.1f}",
                        f"{test_metrics['eval_steps_per_second']:.1f}",
                    ]
                )
            except KeyError as e:
                print(f"[ERROR] {test_metrics} meets {e}")
                continue

        if len(test_metrics_list) > 0:
            print_score_list.append([""])

    custom_header = [
        "exp",
        "ckpt",
        "accuracy",
        "exact_accuracy",
        "lm_loss",
        "q_halt_accuracy",
        "q_halt_loss",
        "steps",
        "runtime",
        "samples_per_second",
        "steps_per_second",
    ]
    pretty_table = list_to_table(print_score_list, custom_header)
    print(pretty_table)

    with open(join(PROJECT_ROOT, "results", "pretty_table.txt"), "w") as f:
        f.write(str(pretty_table))


if __name__ == "__main__":
    main()
