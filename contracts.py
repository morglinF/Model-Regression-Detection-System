"""Interface contract between prompts, classifiers, and the eval pipeline.

The eval pipeline depends only on this module: it gets a PromptConfig from a
PromptLoader, feeds EmailInput to an EmailClassifier, and checks the returned
EmailClassification. Concrete implementations (YAML files, OpenAI) live elsewhere.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["billing", "technical", "account", "general"]


class EmailInput(BaseModel):
    """Input: the raw support email."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    email: str = Field(min_length=1)


class EmailClassification(BaseModel):
    """Output: structured JSON with category and summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Category
    summary: str = Field(description="One-sentence summary of what the customer wants.")


class FewShotExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: EmailInput
    output: EmailClassification


class PromptConfig(BaseModel):
    """A single immutable prompt version — the "code" CI runs against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(pattern=r"^v\d+$")
    created_at: datetime
    description: str = ""
    system_prompt: str = Field(min_length=1)
    user_template: str = Field(min_length=1, description="Must contain {email}.")
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)


class PromptLoader(ABC):
    """Source of versioned PromptConfigs."""

    @abstractmethod
    def list_versions(self, prompt_id: str) -> list[str]:
        """Return available versions, oldest first."""

    @abstractmethod
    def load(self, prompt_id: str, version: Optional[str] = None) -> PromptConfig:
        """Load a specific version, or the latest when version is None."""


class EmailClassifier(ABC):
    """Turns an email into a classification using a given prompt version."""

    @abstractmethod
    def classify(self, email: EmailInput, prompt: PromptConfig) -> EmailClassification:
        ...
