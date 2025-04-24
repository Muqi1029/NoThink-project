import os
from pprint import pprint

from openai import OpenAI

prompt = "肚子疼怎么办？"
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"],
)


outputs = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="gpt-4.1-2025-04-14",
    logprobs=True,
    top_logprobs=3,
)

# used for printing the first logprob
first = True
sum_log_probs = 0
n = 0
for logprob in outputs.choices[0].logprobs.content:
    if first:
        first = False
        pprint(logprob)
        """
        ChatCompletionTokenLogprob(
            token='\\xe8\\x82',
            bytes=[232, 130],
            logprob=-0.0029295608401298523,
            top_logprobs=[
                TopLogprob(token='\\xe8\\x82', bytes=[232, 130], logprob=-0.0029295608401298523),
                TopLogprob(token='你好', bytes=[228, 189, 160, 229, 165, 189], logprob=-6.2529296875),
                TopLogprob(token='很', bytes=[229, 190, 136], logprob=-7.7529296875)
            ]
        )
        """
    sum_log_probs += logprob.logprob
    n += 1
average_log_prob = sum_log_probs / n
print(sum_log_probs, average_log_prob)

first_response = outputs.choices[0].message.content
print(first_response)


prompts = """
This is the logprobs of the first response:
{logprobs}, lower means the response is more creative or has potential of dangerous, please provide the confidence in the answer only in percent (0–100 %) direcly, then following the rationales:
"""

outputs = client.chat.completions.create(
    messages=[
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": prompts.format(logprobs=average_log_prob)},
    ],
    model="gpt-4.1-2025-04-14",
)

print(outputs.choices[0].message.content)
