import argparse
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import DefaultDict, List

from datasets import load_dataset
from sglang import Runtime, assistant, function, gen, select, set_default_backend, user
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class Result:
    question: str
    answer: str
    outputs: DefaultDict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )  # save completion text
    completion_tokens: DefaultDict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )  # save completion tokens
    text: DefaultDict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )  # save context
    choices: DefaultDict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )  # used for gpqa-diamond


def serialize_results(results: list[Result]) -> list[dict]:
    converted = []
    for r in results:
        d = {
            "question": r.question,
            "answer": r.answer,
            "outputs": dict(r.outputs),
            "completion_tokens": dict(r.completion_tokens),
            "text": dict(r.text),
            "choices": dict(r.choices),
        }
        converted.append(d)
    return converted


deepseek_models = ["deepseek-chat", "deepseek-reasoner"]

categories = ["Think", "ThinkOver", "NotThink"]
math_prompt = r"Please reason step by step, and put your final answer within \boxed{}."


def get_ds_api_completion_response(item, client, args) -> tuple[str, int]:
    """
    return completion text and completion tokens

    ERROR - Error: Error code: 400 - {'error': {'message': 'deepseek-reasoner does not support completion api', 'type': '
        invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
    """
    response = client.completions.create(
        model=args.model_path,
        prompt=item["chat_question"],
        max_tokens=args.max_tokens,
    )
    return response.choices[0].text, response.usage.completion_tokens


def get_ds_api_chat_completion_response(item, client, args) -> tuple[str, int]:
    """
    return chat text and chat tokens
    """
    response = client.chat.completions.create(
        model=args.model_path,
        messages=[{"role": "user", "content": item["chat_question"]}],
    )
    return response.choices[0].message.content, response.usage.completion_tokens


# tokenizer for completion
def map_tokenizer(item, tokenizer):
    item["chat_question"] = (
        tokenizer.apply_chat_template(
            [{"content": item["question"], "role": "user"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + "\n</think>"
    )
    return item


################# GSM8K BEGIN ###############################
@function
def gsm_qa(s, item):
    s += user(item["question"] + "\n" + math_prompt)
    forks = s.fork(len(categories))
    for name, fork in zip(categories, forks):
        if name == "Think":
            fork += assistant("<think>" + gen(name))
        elif name == "ThinkOver":
            fork += assistant(
                "<think>I have thought about the problem over</think>" + gen(name)
            )
        elif name == "NotThink":
            fork += assistant("<think>\n</think>" + gen(name))
        else:
            raise ValueError(f"Unknown category: {name}")
        s[name] = fork[name]
        """state.get_meta_info("NotThink")
            {'cached_tokens': 2,
            'completion_tokens': 578,
            'e2e_latency': 17.26391863822937,
            'finish_reason': {'matched': 151643, 'type': 'stop'},
            'id': '238f5780982e4a419c1d525531874b14',
            'prompt_tokens': 115}
        """
        s[f"{name}-counts"] = fork.get_meta_info(name)["completion_tokens"]
        # s[f"{name}-text"] = fork.text()


################# GSM8K END ###############################

################## GPQA DIAMOND BEGIN ############################
gpqa_prompt_template = """What is the correct answer to this question:{Question}
Choices:
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}"""

choices = ["(A)", "(B)", "(C)", "(D)"]


def gpqa_diamond_map_function(item):
    # for preserving the consistency
    item["question"] = item["Question"]
    return item


@function
def gpqa_diamond_qa(s, item):
    s += user(
        gpqa_prompt_template.format(
            Question=item["question"],
            choice1=item["choice1"],
            choice2=item["choice2"],
            choice3=item["choice3"],
            choice4=item["choice4"],
        )
    )
    forks = s.fork(len(categories))
    for name, fork in zip(categories, forks):
        if name == "Think":
            fork += assistant(
                "<think>"
                + gen(name)
                + "So the answer is "
                + select(
                    name=f"{name}-choice",
                    choices=choices,
                    temperature=0.0,
                )
            )
        elif name == "ThinkOver":
            fork += assistant(
                "<think>I have thought about the problem over</think>"
                + gen(name)
                + "So the answer is "
                + select(
                    name=f"{name}-choice",
                    choices=choices,
                    temperature=0.0,
                )
            )
        elif name == "NotThink":
            fork += assistant(
                "<think>\n</think>"
                + gen(name)
                + "So the answer is "
                + select(
                    name=f"{name}-choice",
                    choices=choices,
                    temperature=0.0,
                )
            )
        else:
            raise ValueError(f"Wrong mode {name}")

        s[name] = fork[name]  # store completion text
        s[f"{name}-choice"] = fork[f"{name}-choice"]  # store choice text
        s[f"{name}-counts"] = fork.get_meta_info(name)[
            "completion_tokens"
        ]  # store completion tokens
        s[f"{name}-text"] = fork.text()  # store full text (contain special tokens)


################## GPQA DIAMOND END ############################


################## AIME 2024 BEGIN ############################
def aime_map_function(item):
    item["question"] = item["Problem"]
    item["answer"] = item["Answer"]
    return item


@function
def aime_qa(s, item):
    s += user(f"Problem: {item['question']}\n{math_prompt}")
    forks = s.fork(len(categories))
    for name, fork in zip(categories, forks):
        if name == "Think":
            fork += assistant("<think>\n" + gen(name))
        elif name == "ThinkOver":
            fork += assistant(
                "<think>I have thought about the problem over</think>" + gen(name)
            )
        elif name == "NotThink":
            fork += assistant("<think>\n</think>" + gen(name))
        else:
            raise ValueError(f"Unknown category: {name}")
        s[name] = fork[name]
        s[f"{name}-counts"] = fork.get_meta_info(name)[
            "completion_tokens"
        ]  # store completion tokens
        s[f"{name}-text"] = fork.text()  # store full text (contain special tokens)


################## AIME 2024 END ############################


@function(num_api_spec_tokens=128)
def sgl_hack_deepseek_r1(s, q, mode, choices=None):
    s += user(q)
    if mode == "NotThink":
        if choices is not None:
            s += assistant(
                "<think>\n</think>"
                + gen(mode)
                + "So the answer is "
                + select(name=f"{mode}-choice", choices=choices, temperature=0.0)
            )
        else:
            s += assistant("<think>\n</think>" + gen(mode))
    elif mode == "Think":
        if choices is not None:
            s += assistant(
                "<think>"
                + gen(mode)
                + "So the answer is "
                + select(name=f"{mode}-choice", choices=choices, temperature=0.0)
            )
        else:
            s += assistant("<think>" + gen(mode))
    elif mode == "ThinkOver":
        if choices is not None:
            s += assistant(
                "<think>I have thought about the problem over</think>"
                + gen(mode)
                + "So the answer is "
                + select(name=f"{mode}-choice", choices=choices, temperature=0.0)
            )
        else:
            s += assistant(
                "<think>I have thought about the problem over</think>" + gen(mode)
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# SGLang as a backend
def run_sglang(args, dataset) -> List[Result]:
    logging.info(
        f"Running SGLang with {args.model_path} and {args.batch_size} batch size"
    )

    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    # used to save results
    results = []

    if args.model_path in deepseek_models:
        # use sgl-hack-deepseek-r1
        from sglang import OpenAI

        # use sglang to hack deepseek-r1 using api
        set_default_backend(
            OpenAI(
                model_name=args.model_path,
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ["DEEPSEEK_BASE_URL"],
            )
        )

        # BUG (there is a deadlock when using sglang to hack deepseek-r1)
        # TODO: fix it
        for batch_idx in tqdm(range(num_batches), desc="Running SGLang"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset))

            batch_results = [
                Result(
                    question=dataset[i]["question"],
                    answer=dataset[i]["answer"],
                )
                for i in range(start_idx, end_idx)
            ]
            for mode in categories:
                for _ in range(args.num_samples):
                    # run multiple times for each question
                    if args.dataset == "gpqa-diamond":
                        choices = choices
                    else:
                        choices = None
                    states = sgl_hack_deepseek_r1.run_batch(
                        [
                            {
                                "q": (
                                    dataset[i]["question"] + "\n" + math_prompt
                                    if args.dataset
                                    in ["Maxwell-Jia/AIME_2024", "openai/gsm8k"]
                                    else dataset[i]["question"]
                                ),
                                "mode": mode,
                                "choices": choices,
                            }
                            for i in range(start_idx, end_idx)
                        ],
                    )
                    for i, state in enumerate(states):
                        batch_results[i].outputs[mode].append(state[mode])
                        batch_results[i].completion_tokens[mode].append(
                            state.get_meta_info(mode)["completion_tokens"]
                        )
                        if args.dataset == "gpqa-diamond":
                            batch_results[i].outputs[f"{mode}-choice"].append(
                                state[f"{mode}-choice"]
                            )
            results.extend(batch_results)
    else:
        # run sglang on local model
        set_default_backend(
            Runtime(
                model_path=args.model_path, tp_size=args.tp_size, dp_size=args.dp_size
            )
        )
        if args.dataset == "openai/gsm8k":
            func = gsm_qa
        elif args.dataset == "gpqa-diamond":
            func = gpqa_diamond_qa
        elif args.dataset == "Maxwell-Jia/AIME_2024":
            func = aime_qa
        else:
            raise ValueError(f"Dataset {args.dataset} not supported")

        for cur_batch in tqdm(range(num_batches), desc="Running SGLang"):
            start_idx = cur_batch * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset))

            batch_results = [
                Result(
                    question=dataset[i]["question"],
                    answer=dataset[i]["answer"],
                )
                for i in range(start_idx, end_idx)
            ]

            for _ in range(args.num_samples):  # run multiple times for each question
                states = func.run_batch(
                    [{"item": dataset[i]} for i in range(start_idx, end_idx)],
                    max_new_tokens=args.max_tokens,
                )
                for i, state in enumerate(states):
                    for name in categories:
                        batch_results[i].outputs[name].append(
                            state[name]
                        )  # save completion text
                        batch_results[i].completion_tokens[name].append(
                            state[f"{name}-counts"]
                        )  # save completion tokens
                        batch_results[i].text[name].append(
                            state[f"{name}-text"]
                        )  # save full text
                        if (
                            args.dataset == "gpqa-diamond"
                        ):  # for gpqa-diamond, we need to save choice text
                            batch_results[i].choices[name].append(
                                state[f"{name}-choice"]
                            )

            results.extend(batch_results)
    return results


# DeepSeek API as a backend
def run_deepseek_api(args, dataset) -> List[Result]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/beta",  # this url is only for FIM
        # base_url=os.environ["DEEPSEEK_BASE_URL"], # this base url if for chat competion
    )
    datas = []
    counts = {name: 0 for name in categories}
    invalid_count = 0
    with tqdm(total=len(dataset), desc="Running DeepSeek API") as pbar:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(get_ds_api_completion_response, item, client, args)
                # executor.submit(get_ds_api_chat_completion_response, item, client, args)
                for item in dataset
            ]
            for item, future in zip(dataset, futures):
                try:
                    response, response_tokens = future.result()
                    datas.append(
                        {
                            "question": item["question"],
                            "chat_question": item["chat_question"],
                            "answer": item["answer"],
                            "output": response,
                        }
                    )
                    counts["Think"] += response_tokens
                except Exception as e:
                    logging.error(f"Error: {e}")
                    invalid_count += 1
                    continue
                finally:
                    pbar.update(1)
    # TODO: return invalid_count
    return datas, counts


def main(args):
    if args.dataset == "openai/gsm8k":
        dataset = load_dataset(args.dataset, name="main", split="test")
    elif args.dataset == "gpqa-diamond":
        dataset = load_dataset(
            "json",
            data_files="data/gpqa_diamond.json",
            split="train",
        )
        dataset = dataset.map(gpqa_diamond_map_function, batched=False)
    elif args.dataset == "Maxwell-Jia/AIME_2024":
        dataset = load_dataset(args.dataset, split="train")
        dataset = dataset.map(aime_map_function, batched=False)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.debug:
        logging.info(f"Debug mode, only use 4 examples for testing")
        dataset = dataset.select(range(4))

    model_name = args.model_path.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]

    if args.backend == "ds-api":
        # ds-api only support deepseek-chat and deepseek-reasoner
        assert args.model_path in [
            "deepseek-chat",
            "deepseek-reasoner",
        ], f"ds-api only support deepseek-chat and deepseek-reasoner, but got {args.model_path}"
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        dataset = dataset.map(
            partial(map_tokenizer, tokenizer=tokenizer), batched=False
        )

        # info
        logging.info("-" * 100)
        logging.info(dataset["question"][:3])
        logging.info(dataset["chat_question"][:3])
        logging.info("-" * 100)
        data, count = run_deepseek_api(args, dataset)
    elif args.backend == "sglang":
        results = run_sglang(args, dataset)
    else:
        raise ValueError(f"Backend {args.backend} not supported")

    os.makedirs("results", exist_ok=True)
    file_name = (
        f"{args.backend}_{model_name}_{dataset_name}_{datetime.now().strftime('%02m%02d')}.json"
        if not args.debug
        else f"{args.backend}-debug_{model_name}_{dataset_name}_{datetime.now().strftime('%02m%02d')}.json"
    )
    with open(f"results/{file_name}", "w") as f:
        json.dump(serialize_results(results), f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument(
        "--backend",
        choices=["ds-api", "sglang"],
        default="sglang",
        help="Backend to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["openai/gsm8k", "gpqa-diamond", "Maxwell-Jia/AIME_2024"],
        help="Dataset to use",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--max-tokens", type=int, default=32768, help="Max tokens to generate"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size")
    parser.add_argument("--dp-size", type=int, default=1, help="DP size")
    parser.add_argument("--top-p", type=float, default=0.95, help="top p size")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Only used for ds-api in concurrent mode",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples for each question",
    )

    args = parser.parse_args()
    print(args)
    main(args)
