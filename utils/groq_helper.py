"""
Wraps calls to the Groq API (OpenAI-compatible chat completions).
Get a free key from https://console.groq.com/keys
"""

from groq import Groq

# Groq's currently supported fast, high-quality general model.
# Change this if Groq deprecates/renames the model.
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def generate_ai_summary(api_key: str, dataset_context: str, model: str = DEFAULT_MODEL) -> str:
    """One-shot automatic 'what does this data tell us' summary."""
    client = get_client(api_key)

    system_prompt = (
        "You are a senior data analyst. You will be given a compact statistical "
        "summary of a tabular dataset (not the raw data). Write a clear, structured "
        "EDA insight report in markdown covering: (1) what the data appears to represent, "
        "(2) data quality issues worth flagging, (3) notable patterns or relationships, "
        "(4) 3-5 concrete next-step recommendations. Be specific, reference actual column "
        "names and numbers from the summary, and keep it under 350 words."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dataset_context},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return response.choices[0].message.content


def chat_about_dataset(api_key: str, dataset_context: str, chat_history: list, user_question: str,
                        model: str = DEFAULT_MODEL) -> str:
    """
    Multi-turn Q&A. chat_history is a list of {"role": "user"/"assistant", "content": str}
    from previous turns (excluding the system prompt).
    """
    client = get_client(api_key)

    system_prompt = (
        "You are a helpful data analysis assistant. The user has uploaded a dataset. "
        "You only have access to the statistical summary below, not the raw rows, so "
        "answer based on that summary and general data science reasoning. If asked "
        "something the summary can't answer, say so plainly rather than guessing.  \n\n"
        f"DATASET SUMMARY:\n{dataset_context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_question})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=700,
    )
    return response.choices[0].message.content
