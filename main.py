import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

from datasets import load_dataset
from sglang import Runtime, assistant, function, gen, select, set_default_backend, user
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

deepseek_models = ["deepseek-chat", "deepseek-reasoner"]

categories = ["Think", "ThinkOver", "NotThink"]
math_prompt = (
    "Please reason step by step, and put your final answer within \\boxed\{\}."
)


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


#################  GSM8K ###############################
@function
def gsm_qa(s, item):
    s += user(item["question"])
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
        s[f"{name}-text"] = fork.text()


################## GPQA DIAMOND ############################
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


@function(num_api_spec_tokens=128)
def sgl_hack_deepseek_r1(s, q):
    s += user(q)
    s += assistant("<think>\n</think>" + gen("NotThink"))


# SGLang as a backend
def run_sglang(args, dataset):
    logging.info(
        f"Running SGLang with {args.model_path} and {args.batch_size} batch size"
    )

    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    # used to save results
    data = []
    counts = {name: [] for name in categories}

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

        # BUG (there is deadlock when using sglang to hack deepseek-r1)
        # TODO: fix it
        for batch_idx in tqdm(range(num_batches), desc="Running SGLang"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset))
            states = sgl_hack_deepseek_r1.run_batch(
                [{"q": dataset[i]["question"]} for i in range(start_idx, end_idx)],
                # max_new_tokens=args.max_tokens, # can not be set due to api_spec_tokens
            )
            for i, state in enumerate(states):
                print(state.get_meta_info("NotThink"))
                data.append(
                    {
                        "question": dataset[start_idx + i]["question"],
                        "answer": dataset[start_idx + i]["answer"],
                        "NotThink-text": state.text(),
                        "NotThink": state["NotThink"],
                    }
                )
            return data, counts

    else:
        # run sglang on local model
        set_default_backend(
            Runtime(
                model_path=args.model_path, tp_size=args.tp_size, dp_size=args.dp_size
            )
        )
        for cur_batch in tqdm(range(num_batches), desc="Running SGLang"):
            start_idx = cur_batch * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset))

            if args.dataset == "openai/gsm8k":
                func = gsm_qa
            elif args.dataset == "gpqa-diamond":
                func = gpqa_diamond_qa
            else:
                raise ValueError(f"Dataset {args.dataset} not supported")

            states = func.run_batch(
                [{"item": dataset[i]} for i in range(start_idx, end_idx)],
                max_new_tokens=args.max_tokens,
            )

            for i, state in enumerate(states):
                item = {
                    "question": dataset[start_idx + i]["question"],
                    "answer": dataset[start_idx + i]["answer"],
                }
                for name in categories:
                    # save full text and completion text
                    item[f"{name}-text"] = state[f"{name}-text"]
                    item[name] = state[name]
                    if args.dataset == "gpqa-diamond":
                        # for gpqa-diamond, we need to save choice text
                        item[f"{name}-choice"] = state[f"{name}-choice"]
                data.append(item)

                # save counts
                for name in categories:
                    counts[name].append(state[f"{name}-counts"])

        for name in categories:
            # recompute counts (mean, ...)
            counts[f"{name}-MEAN(tokens)"] = sum(counts[name]) / len(counts[name])
            del counts[name]

        return data, counts


# DeepSeek API as a backend
def run_deepseek_api(args, dataset):
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
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.debug:
        logging.info(f"Debug mode, only use 5 examples for testing")
        dataset = dataset.select(range(5))

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
        data, count = run_sglang(args, dataset)
    else:
        raise ValueError(f"Backend {args.backend} not supported")

    os.makedirs("results", exist_ok=True)
    with open(
        f"results/{args.backend}_{model_name}_{dataset_name}_{datetime.now().strftime('%02m%02d')}.json",
        "w",
    ) as f:
        json.dump(
            {
                "count": count,
                "data": data,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


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
        choices=["openai/gsm8k", "gpqa-diamond"],
        help="Dataset to use",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--max-tokens", type=int, default=32000, help="Max tokens to generate"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size")
    parser.add_argument("--dp-size", type=int, default=1, help="DP size")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Only used for ds-api in concurrent mode",
    )

    args = parser.parse_args()
    main(args)
