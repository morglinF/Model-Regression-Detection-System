# Model-Regression-Detection-System
CI/CD pipeline that continuously tests any LLM-powered feature against a golden dataset whenever a prompt or model changes, detects quality regressions, and alerts your team via Slack before bad outputs reach users.

> **Status:** the feature under test (a support-email classifier) and versioned prompts are in place. The golden dataset, eval runner, CI workflow, and Slack alerts are not built yet — see [Roadmap](#roadmap).

## The feature under test

A customer support email classifier.

- **Input:** email text
- **Output:** structured JSON — `{"category": "billing" | "technical" | "account" | "general", "summary": "<one sentence>"}`

## How it works

```
prompts/email_classifier/v1.yaml ──► YamlPromptLoader ──► PromptConfig ─┐
                                                                        ├──► OpenAIEmailClassifier ──► EmailClassification
                                             email text ──► EmailInput ─┘
```

1. `YamlPromptLoader` reads a prompt version from `prompts/` and validates it into a `PromptConfig`.
2. The email is wrapped in an `EmailInput`.
3. `OpenAIEmailClassifier` renders the messages (system prompt → few-shot user/assistant pairs → the email) and calls OpenAI with [structured outputs](https://platform.openai.com/docs/guides/structured-outputs), so the response is schema-enforced and parsed into an `EmailClassification`.

Every result is attributable to an exact prompt version and model — that pair is what the eval pipeline will compare against a baseline.

## Project layout

| File | Purpose |
|---|---|
| `contracts.py` | The interface contract (Pydantic): `EmailInput`, `EmailClassification`, `PromptConfig`, `FewShotExample`, and the `PromptLoader` / `EmailClassifier` interfaces. The eval pipeline depends only on this. |
| `prompts/<prompt_id>/vN.yaml` | Versioned prompts — the "code" CI runs against. |
| `prompt_loader.py` | `YamlPromptLoader`: lists versions, loads latest or a pinned version, validates files. |
| `email_classifier.py` | `OpenAIEmailClassifier`: OpenAI implementation of `EmailClassifier`. Runnable as a demo. |
| `.env.example` | Template for required environment variables. |

## Versioned prompts

Each prompt version is an immutable YAML file:

```yaml
id: email_classifier
version: v1
created_at: "2026-09-27T00:00:00Z"
description: Baseline support-email triage prompt (zero-shot).
system_prompt: |
  You are a customer support email triage assistant. ...
user_template: |
  Email:
  {email}
few_shot_examples:
  - input: {email: "I can't log in to my account."}
    output: {category: account, summary: "Customer needs help logging in."}
```

Loading validates that required fields are present, `version` matches `vN`, the `id`/`version` match the file's location, and `user_template` contains `{email}`.

**To change a prompt:** copy the latest file to the next version (e.g. `v2.yaml`), bump `version` and `created_at`, and edit. Never modify a published version — past eval results must stay reproducible.

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # then set OPENAI_API_KEY
```

`.env` is git-ignored; never commit it.

## Usage

Run the demo:

```bash
python email_classifier.py
# [email_classifier@v1] {"category":"general","summary":"..."}
```

From code:

```python
from contracts import EmailInput
from prompt_loader import YamlPromptLoader
from email_classifier import OpenAIEmailClassifier

prompt = YamlPromptLoader().load("email_classifier", "v1")   # omit version for latest
classifier = OpenAIEmailClassifier(model="gpt-4o-mini")      # default: gpt-4o
result = classifier.classify(EmailInput(email="I was charged twice."), prompt)
print(result.category, result.summary)
```

## Roadmap

- [x] Email classifier feature
- [x] Versioned prompts in YAML
- [x] Typed interface contract (Pydantic)
- [ ] Golden dataset of labeled emails
- [ ] Eval runner: score a prompt version + model against the dataset and compare to a baseline
- [ ] CI workflow triggered on prompt/model changes
- [ ] Slack alerts on regressions
