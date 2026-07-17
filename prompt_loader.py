def load_prompt():

    with open(
        "prompt/prompt.txt",
        "r",
        encoding="utf-8"
    ) as file:

        prompt = file.read()

    return prompt