import json
import os
import re


import backoff
import openai

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
]


# Get N responses from a single message, used for ensembling.
# @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
# def get_batch_responses_from_llm(
#         msg,
#         client,
#         model,
#         system_message,
#         print_debug=False,
#         msg_history=None,
#         temperature=0.75,
#         n_responses=1,
# ):
#     if msg_history is None:
#         msg_history = []

#     if 'gpt' in model:
#         new_msg_history = msg_history + [{"role": "user", "content": msg}]
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": system_message},
#                 *new_msg_history,
#             ],
#             temperature=temperature,
#             max_tokens=MAX_NUM_TOKENS,
#             n=n_responses,
#             stop=None,
#             seed=0,
#         )
#         content = [r.message.content for r in response.choices]
#         new_msg_history = [
#             new_msg_history + [{"role": "assistant", "content": c}] for c in content
#         ]
#     elif model == "llama-3-1-405b-instruct":
#         new_msg_history = msg_history + [{"role": "user", "content": msg}]
#         response = client.chat.completions.create(
#             model="meta-llama/llama-3.1-405b-instruct",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 *new_msg_history,
#             ],
#             temperature=temperature,
#             max_tokens=MAX_NUM_TOKENS,
#             n=n_responses,
#             stop=None,
#         )
#         content = [r.message.content for r in response.choices]
#         new_msg_history = [
#             new_msg_history + [{"role": "assistant", "content": c}] for c in content
#         ]
#     else:
#         content, new_msg_history = [], []
#         for _ in range(n_responses):
#             c, hist = get_response_from_llm(
#                 msg,
#                 client,
#                 model,
#                 system_message,
#                 print_debug=False,
#                 msg_history=None,
#                 temperature=temperature,
#             )
#             content.append(c)
#             new_msg_history.append(hist)

#     if print_debug:
#         print()
#         print("*" * 20 + " LLM START " + "*" * 20)
#         for j, msg in enumerate(new_msg_history[0]):
#             print(f'{j}, {msg["role"]}: {msg["content"]}')
#         print(content)
#         print("*" * 21 + " LLM END " + "*" * 21)
#         print()

#     return content, new_msg_history


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    # if "claude" in model:
    #     new_msg_history = msg_history + [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": msg,
    #                 }
    #             ],
    #         }
    #     ]
    #     response = client.messages.create(
    #         model=model,
    #         max_tokens=MAX_NUM_TOKENS,
    #         temperature=temperature,
    #         system=system_message,
    #         messages=new_msg_history,
    #     )
    #     content = response.content[0].text
    #     new_msg_history = new_msg_history + [
    #         {
    #             "role": "assistant",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": content,
    #                 }
    #             ],
    #         }
    #     ]
    if 'gpt' in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "o1" in model or "o3" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            max_completion_tokens=MAX_NUM_TOKENS,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    # elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
    #     new_msg_history = msg_history + [{"role": "user", "content": msg}]
    #     response = client.chat.completions.create(
    #         model="meta-llama/llama-3.1-405b-instruct",
    #         messages=[
    #             {"role": "system", "content": system_message},
    #             *new_msg_history,
    #         ],
    #         temperature=temperature,
    #         max_tokens=MAX_NUM_TOKENS,
    #         n=1,
    #         stop=None,
    #     )
    #     content = response.choices[0].message.content
    #     new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    # elif model in ["deepseek-chat", "deepseek-coder"]:
    #     new_msg_history = msg_history + [{"role": "user", "content": msg}]
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_message},
    #             *new_msg_history,
    #         ],
    #         temperature=temperature,
    #         max_tokens=MAX_NUM_TOKENS,
    #         n=1,
    #         stop=None,
    #     )
    #     content = response.choices[0].message.content
    #     new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    # elif model in ["deepseek-reasoner"]:
    #     new_msg_history = msg_history + [{"role": "user", "content": msg}]
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_message},
    #             *new_msg_history,
    #         ],
    #         n=1,
    #         stop=None,
    #     )
    #     content = response.choices[0].message.content
    #     new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    # elif "gemini" in model:
    #     new_msg_history = msg_history + [{"role": "user", "content": msg}]
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_message},
    #             *new_msg_history,
    #         ],
    #         temperature=temperature,
    #         max_tokens=MAX_NUM_TOKENS,
    #         n=1,
    #     )
    #     content = response.choices[0].message.content
    #     new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found

