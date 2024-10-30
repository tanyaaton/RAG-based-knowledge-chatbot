def prompt_EN(question, context):
    messages = {
        "messages": [
            {
            "content": "You are a helpful, respectful assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.You will receive Document on user queries DOCUMENT, and QUESTION from user below. Answer the question.",
            "role": "system"
            },
            {
            "content": f"""DOCUMENT: {context}
QUESTION: {question}""",
            "role": "user"
            },
            {
            "content": "Answer: {",
            "role": "assistant"
            },
        ]
    }
    return messages["messages"]