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


def get_index_from_name(name: str):
    if "checkpoint-" in name:
        index = int(basename(dirname(name)).split("-")[-1])
    elif "_ckpt" in name:
        index = 1e8 + int(basename(dirname(name)).split("_ckpt")[-1])
    else:
        raise ValueError(f"Unrecognized name: {name}")
    return index


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

        glob_pattern = join(exp_folder_path, "*/arc_metrics.json")
        if args.compile:
            glob_pattern = glob_pattern.replace(".json", "_compile.json")
        test_metrics_list = glob.glob(glob_pattern)

        if len(test_metrics_list) == 0:
            continue

        test_metrics_list.sort(key=get_index_from_name)

        for test_metrics_path in test_metrics_list:
            with open(test_metrics_path, "r") as f:
                test_metrics = json.load(f)

            ckpt_name = basename(dirname(test_metrics_path))
            exp_name = basename(dirname(dirname(test_metrics_path)))

            try:
                print_score_list.append(
                    [
                        exp_name,
                        ckpt_name,
                        f"{test_metrics['ARC/pass@1']:.2%}",
                        f"{test_metrics['ARC/pass@2']:.2%}",
                        f"{test_metrics['ARC/pass@5']:.2%}",
                        f"{test_metrics['ARC/pass@10']:.2%}",
                        f"{test_metrics['ARC/pass@100']:.2%}",
                        f"{test_metrics['ARC/pass@1000']:.2%}",
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
        "pass@1",
        "pass@2",
        "pass@5",
        "pass@10",
        "pass@100",
        "pass@1000",
    ]
    pretty_table = list_to_table(print_score_list, custom_header)
    print(pretty_table)

    with open(join(PROJECT_ROOT, "results", "arc_pretty_table.txt"), "w") as f:
        f.write(str(pretty_table))


if __name__ == "__main__":
    main()
