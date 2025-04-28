# Dev Road to No-Think Project

## Result Structure

Assuming we have 3 samples, the result structure is as follows:

```json
{
    "question": "question",
    "answer": "answer",
    "outputs": {
        "NotThink": [
            "text1",
            "text2",
            "text3"
        ],
        "NotThinkOver": [
            "text1",
            "text2",
            "text3"
        ],
        "Think": [
            "text1",
            "text2",
            "text3"
        ],
    "completion_tokens": {
        "NotThink": [100, 200, 300],
        "Think": [100, 200, 300],
        "ThinkOver": [100, 200, 300]
    },
	

}
```

## Metrics

### Pass@K
