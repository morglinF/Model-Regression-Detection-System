"""Customer support email classifier powered by OpenAI."""

import os
from typing import Optional

import openai
from dotenv import load_dotenv

from contracts import EmailClassification, EmailClassifier, EmailInput, PromptConfig
from prompt_loader import YamlPromptLoader

load_dotenv()

PROMPT_ID = "email_classifier"


def get_client() -> openai.OpenAI:
    """Build an OpenAI client using OPENAI_API_KEY from the environment / .env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root "
            "(see .env.example)."
        )
    return openai.OpenAI(api_key=api_key)


def build_messages(prompt: PromptConfig, email: EmailInput) -> list[dict]:
    """Render system prompt, few-shot user/assistant turns, then the real email."""

    def render(e: EmailInput) -> str:
        # Plain replacement rather than str.format, so literal braces don't break it.
        return prompt.user_template.replace("{email}", e.email).strip()

    messages = [{"role": "system", "content": prompt.system_prompt.strip()}]
    for example in prompt.few_shot_examples:
        messages.append({"role": "user", "content": render(example.input)})
        messages.append({"role": "assistant", "content": example.output.model_dump_json()})
    messages.append({"role": "user", "content": render(email)})
    return messages


class OpenAIEmailClassifier(EmailClassifier):
    def __init__(
        self,
        model: str = "gpt-4o",  # or "gpt-4o-mini" for a cheaper/faster pass
        client: Optional[openai.OpenAI] = None,
    ):
        self.model = model
        self.client = client or get_client()

    def classify(self, email: EmailInput, prompt: PromptConfig) -> EmailClassification:
        # Structured outputs: the API enforces EmailClassification's JSON schema
        # and the SDK parses the reply back into the Pydantic model.
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=build_messages(prompt, email),
            response_format=EmailClassification,
        )
        message = response.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"Model returned no parsed output (refusal: {message.refusal!r})")
        return message.parsed


if __name__ == "__main__":
    email_text = """Hello,
I want to know more about your products."""
    prompt = YamlPromptLoader().load(PROMPT_ID)
    result = OpenAIEmailClassifier().classify(EmailInput(email=email_text), prompt)
    print(f"[{prompt.id}@{prompt.version}]", result.model_dump_json())
