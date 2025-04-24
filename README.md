# Project Structure

```bash
.
├── eval.py
├── main.py
├── data # save the dataset
├── README.md
├── results
```

- main.py: The main script run the model inference.
  - `--backend`: The backend to use, currently `sglang` or `ds-api` is supported.
  - `--model`: The model to use, currently `DeepSeek-R1-Distill-Qwen-7B` is supported.
  - `--dataset`: The dataset to use, currently `gsm8k`, `gpqa-diamond`
- eval.py: The script run the evaluation, compute the metrics
- results: The directory to save the results.

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
- [ ] AIME (free-form QA)
- [ ] ADD pass@k as a metric

> settings:
>
> - `max_tokens`: 2048
> - `temperature`: default (We don't set it)

## GSM8k (free-form QA)

| Model                       | Mode      | Accuracy(%) | INVALID_COUNT | Completion Tokens(Avg) |
| --------------------------- | --------- | ----------- | ------------- | ---------------------- |
| DeepSeek-R1-Distill-Qwen-7B | Think     | 79.75       | 0             | 959.87                 |
| DeepSeek-R1-Distill-Qwen-7B | ThinkOver | 76.49       | 2             | 331.80                 |
| DeepSeek-R1-Distill-Qwen-7B | NotThink  | 79.37       | 3             | 292.79                 |

> Explanation:
>
> - `INVALID_COUNT`: The number of invalid answers (which we cannot extract the final number)

## GPQA-Diamond (choices)

| Model                       | Mode      | Accuracy(%) | Completion Tokens(Avg) |
| --------------------------- | --------- | ----------- | ---------------------- |
| DeepSeek-R1-Distill-Qwen-7B | Think     | 34.85       | 1994.01                |
| DeepSeek-R1-Distill-Qwen-7B | ThinkOver | 44.95       | 1047.34                |
| DeepSeek-R1-Distill-Qwen-7B | NotThink  | 38.38       | 633.38                 |

> Explanation:
>
> - We use SGLang `choice` primitive to select the answer from the choices. So the `INVALID_COUNT` is 0.

## AIME
