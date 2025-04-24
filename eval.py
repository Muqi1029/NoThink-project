import json
import re
from collections import defaultdict
from pprint import pprint
from typing import Dict, List, Union

from main import categories


def extract_answer_math(answer: str) -> str:
    return re.findall(r"\d[\.\d]", answer)[-1]


def trim_result_gsm8k(output: str) -> Union[str, int]:
    """tranfer an output string to an exact number"""
    # replace numbers like `x,xxx` with `xxxx`
    output = re.sub(r"(\d),(\d)", r"\1\2", output)

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)

    return numbers[-1] if numbers else None


def check_args(dataset, f_path):
    assert (
        dataset is not None or f_path is not None
    ), "Either dataset or f_path must be provided"
    assert (
        dataset is None or f_path is None
    ), "Only one of dataset or f_path can be provided"
    return True


def eval(name: str, dataset: List[Dict] = None, f_path: str = None):
    assert check_args(
        dataset, f_path
    ), f"Invalid arguments, dataset: {dataset}, f_path: {f_path}"

    if f_path is not None:
        with open(f_path, "r") as f:
            dataset = json.load(f)
        return eval(name, dataset)

    metrics = defaultdict(float)
    count = 0

    if dataset is not None:
        for category in categories:
            metrics[f"{category}-MEAN(tokens)"] = dataset["count"][
                f"{category}-MEAN(tokens)"
            ]

        for item in dataset["data"]:
            for category in categories:
                if name == "gsm8k":
                    pred = trim_result_gsm8k(item[category])
                    gt = trim_result_gsm8k(item["answer"])
                elif name == "gpqa_diamond":
                    pred = item[f"{category}-choice"]
                    gt = item["answer"]
                else:
                    raise ValueError(f"Invalid dataset name: {name}")

                if pred == None:
                    metrics[f"{category}_invalid"] += 1
                    continue
                if str(pred) == str(gt):
                    metrics[category] += 1
            count += 1
        # compute accuracy
        for category in categories:
            metrics[f"{category}_accuracy"] = metrics[category] / count
            del metrics[category]
    return metrics


if __name__ == "__main__":
    print("Evaluating gsm8k".center(100, "-"))
    metrics = eval(
        name="gsm8k",
        f_path="results/sglang_DeepSeek-R1-Distill-Qwen-7B_gsm8k_0424.json",
    )
    pprint(metrics)

    print("Evaluating gpqa_diamond".center(100, "-"))
    metrics = eval(
        name="gpqa_diamond",
        f_path="results/sglang_DeepSeek-R1-Distill-Qwen-7B_gpqa-diamond_0424.json",
    )
    pprint(metrics)
