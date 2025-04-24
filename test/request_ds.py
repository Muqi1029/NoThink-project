import os

from openai import OpenAI

model = "deepseek-reasoner"


def get_ds_api_completion_response(item, client) -> tuple[str, int]:
    """
    return completion text and completion tokens
    """
    response = client.completions.create(
        model=model,
        prompt=item["chat_question"],
    )
    return response.choices[0].text, response.usage.completion_tokens


def get_ds_api_chat_completion_response(item, client) -> tuple[str, int]:
    """
    return chat text and chat tokens
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": item["chat_question"]}],
    )
    return response.choices[0].message.content, response.usage.completion_tokens


if __name__ == "__main__":
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url=os.environ["DEEPSEEK_BASE_URL"]
    )
    item = {
        "chat_question": "What is the capital of France?",
    }
    print(get_ds_api_completion_response(item, client))
    print(get_ds_api_chat_completion_response(item, client))
