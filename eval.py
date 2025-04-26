import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from glob import glob
from pprint import pprint
from typing import Dict, List, Union

import numpy as np

from main import Result, categories


@dataclass
class EvalMetric:
    category: str
    pass_at_k: List[List[float]] = field(default_factory=list)
    completion_tokens: List[List[int]] = field(default_factory=list)


def extract_answer_math(answer: str) -> str:
    return re.findall(r"\d[\.\d]", answer)[-1]


def extract_answer(s: str) -> str:
    pred = ""
    if "boxed" in s:
        ans = s.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a

    pred = re.sub(r"\n\s*", "", pred)
    if pred != "":
        if pred[0] == ":":
            pred = pred[1:]
        if pred[-1] in "./":
            pred = pred[:-1]
    return pred


def deserialize_results(data: list[dict]) -> list[Result]:
    results = []
    for d in data:
        result = Result(
            question=d["question"],
            answer=d["answer"],
            outputs=defaultdict(list, d.get("outputs", {})),
            completion_tokens=defaultdict(list, d.get("completion_tokens", {})),
        )
        results.append(result)
    return results


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


def parallel_eval(ground_truth: str, pred: str) -> bool:
    return extract_answer(pred) == str(ground_truth)


def compute_pass_at_k(is_correct_results: List[bool], k: int) -> float:
    n = len(is_correct_results)
    assert (
        n > 0 and k > 0 and k <= n
    ), f"n and k must be greater than 0, k must be less than or equal to n, but got n={n} and k={k}"

    c = sum(is_correct_results)
    if c == 0:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def eval_pass_at_k(
    dataset_name: str, ks: List[int], dataset: List[Dict] = None, f_path: str = None
):
    assert check_args(
        dataset, f_path
    ), f"Invalid arguments, dataset: {dataset}, f_path: {f_path}"

    if f_path is not None:
        with open(f_path, "r") as f:
            dataset = json.load(f)

    num_samples = len(dataset[0]["outputs"][categories[0]])
    if any(k > num_samples for k in ks):
        raise ValueError(
            f"k is greater than the number of samples: {num_samples}, k={ks}"
        )

    metrics = {category: EvalMetric(category) for category in categories}
    for result_item in dataset:
        if dataset_name == "gsm8k":
            ground_truth = result_item["answer"].split("####")[1].strip()
        elif dataset_name == "AIME_2024":
            ground_truth = result_item["answer"]
        elif dataset_name == "gpqa-diamond":
            ground_truth = result_item["answer"]
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        # compute pass@k for each category
        for category, l in result_item["outputs"].items():
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(parallel_eval, ground_truth, pred) for pred in l
                ]
                is_correct_results = [future.result() for future in futures]
                pass_at_k = [compute_pass_at_k(is_correct_results, k) for k in ks]
                metrics[category].pass_at_k.append(pass_at_k)

        # compute completion tokens for each category
        for category, l in result_item["completion_tokens"].items():
            metrics[category].completion_tokens.append(l)

    # compute mean of pass@k and completion tokens for each category
    # TODO: maybe we can compute other metrics like 99% percentile, std, etc. here

    output_metrics = {}
    for category in categories:
        output_metrics[f"{category}-MEAN(pass@k)"] = np.mean(
            metrics[category].pass_at_k, axis=0
        )  # return a list of mean values

        output_metrics[f"{category}-MEAN(tokens)"] = np.mean(
            metrics[category].completion_tokens
        )  # return a float
    return output_metrics


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


def main():
    # datasets = ["gsm8k", "gpqa-diamond", "AIME_2024"]
    datasets = ["gsm8k"]
    ks = [1, 5]
    model_name = "DeepSeek-R1-Distill-Qwen-7B"

    for dataset_name in datasets:
        # find the latest file on default
        files = glob(f"results/*{model_name}*{dataset_name}*.json")
        if len(files) == 0:
            continue

        print(f"Evaluating {dataset_name}".center(100, "-"))

        # find the latest file on default
        latest_file = max(
            files, key=lambda x: int(re.search(r"_(\d{4})\.json$", x).group(1))
        )
        print(f"Using {latest_file} as the latest file")

        metrics = eval_pass_at_k(
            dataset_name=dataset_name,
            ks=ks,
            f_path=latest_file,
        )

        # make markdown table
        table = (
            "\n| Model | Category | "
            + " | ".join([f"MEAN(pass@{k})" for k in ks])
            + " | MEAN(tokens) |"
        )
        table += "\n| --- | --- | " + " | ".join(["---"] * (len(ks) + 1)) + " |\n"
        for category in categories:
            table += (
                f"| {model_name} | {category} | "
                + " | ".join([f"{x:.4f}" for x in metrics[f"{category}-MEAN(pass@k)"]])
                + " | "
                + f"{metrics[f'{category}-MEAN(tokens)']:.4f}"
                + " |\n"
            )
        print(table)


if __name__ == "__main__":
    main()

    # print("Evaluating gsm8k".center(100, "-"))
    # metrics = eval(
    #     name="gsm8k",
    #     f_path="results/sglang_DeepSeek-R1-Distill-Qwen-7B_gsm8k_0424.json",
    # )
    # pprint(metrics)

    # print("Evaluating gpqa_diamond".center(100, "-"))
    # metrics = eval(
    #     name="gpqa_diamond",
    #     f_path="results/sglang_DeepSeek-R1-Distill-Qwen-32B_gpqa-diamond_0425.json",
    # )
    # pprint(metrics)
