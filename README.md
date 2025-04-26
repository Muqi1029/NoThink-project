# Project Structure

```bash
.
├── eval.py
├── main.py
├── data # save the dataset
├── README.md
```

- main.py: The main script run the model inference.
  - `--backend`: The backend to use, currently `sglang` or `ds-api` is supported.
  - `--model`: The model to use, currently `DeepSeek-R1-Distill-Qwen-7B` is supported.
  - `--dataset`: The dataset to use, currently `gsm8k`, `gpqa-diamond`, `AIME_2024` is supported.
- eval.py: The script run the evaluation, compute the metrics

# Mode Explanation

- `Think`(baseline): The model generally thinks about the question and then answers it.

```python
<assistant><think>...
```

- `ThinkOver`: The model thinks about the question and then answers it.

```python
<assistant><think>I have thought about the problem over</think>...
```

- `NotThink`: The model does not think about the question and just answers it.

```python
<assistant><think>\n</think>...
```

# Benchmarks

Objectives:

- [x] GSM8k (free-form QA)
- [x] GPQA-Diamond (choices)
- [x] AIME (free-form QA)
- [x] ADD pass@k as a metric

> settings:
>
> - `max_tokens`: 32,000
> - `temperature`: default (In terms of DeepSeek-R1-Distill-Qwen-7B, it's set as 0.6)

## GSM8k (free-form QA)

| Model | Category | MEAN(pass@1) | MEAN(pass@5) | MEAN(tokens) |
| --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | Think | 0.9318 | 0.9698 | 2463.3235 |
| DeepSeek-R1-Distill-Qwen-7B | ThinkOver | 0.6184 | 0.9542 | 393.2460 |
| DeepSeek-R1-Distill-Qwen-7B | NotThink | 0.8290 | 0.9639 | 289.5554 |

## GPQA-Diamond (choices)

| Model                       | Mode      | Accuracy(%) | Completion Tokens(Avg) |
| --------------------------- | --------- | ----------- | ---------------------- |
| DeepSeek-R1-Distill-Qwen-7B | Think     | 47.98       | 19827.53|
| DeepSeek-R1-Distill-Qwen-7B | ThinkOver | 42.42       | 2500.12|
| DeepSeek-R1-Distill-Qwen-7B | NotThink  | 39.39       | 768.45|

> Explanation:
>
> - We use SGLang `choice` primitive to select the answer from the choices.

## AIME

| Model | Category | MEAN(pass@1) | MEAN(pass@5) | MEAN(tokens) |
| --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | Think | 0.4600 | 0.6857 | 9934.0333 |
| DeepSeek-R1-Distill-Qwen-7B | ThinkOver | 0.4267 | 0.6878 | 9963.3867 |
| DeepSeek-R1-Distill-Qwen-7B | NotThink | 0.2133 | 0.5185 | 3664.4733 |

-- --
