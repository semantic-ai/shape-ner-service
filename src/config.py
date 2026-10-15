"""
Configuration Management with Pydantic Validation

This module provides Pydantic models for validating application configuration
loaded from config.json.
"""

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from decide_ai_service_base.config import load_config


class LlmConfig(BaseModel):
    """LLM provider configuration for the segmentation task."""

    provider: str = Field(
        default="ollama",
        description="LangChain provider name, e.g. 'ollama', 'openai', 'mistral'"
    )
    model_name: str = Field(
        default="mistral-nemo",
        description="Model name as understood by the provider"
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the provider"
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL (required for Ollama and self-hosted endpoints)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Generation temperature (lower = more deterministic)"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Max attempts on LLM call failure"
    )
    retry_delay: float = Field(
        default=15.0,
        ge=0,
        description="Seconds to sleep between retries"
    )


class TargetConfig(BaseModel):
    """
    A kind of task input and how to segment it, after the targets of the annotation-review-service: which inputs it
    applies to (a SPARQL check on ?target) and the SHACL node shapes to segment them along. Relative paths are
    resolved against the project root.
    """

    label: str | None = Field(
        default=None,
        description="Human-readable name of the target"
    )
    prefixes: str = Field(
        default="",
        description="PREFIX declarations used by target_filter"
    )
    target_filter: str = Field(
        description="SPARQL graph pattern on ?target, a resource in the task's input container: the target applies "
                    "to the resources it matches, e.g. '?target a eli:Expression .'"
    )
    text_predicate: str = Field(
        description="Property of the input resource holding the text to segment"
    )
    shape_path: str = Field(
        description="SHACL shape (Turtle) describing what to segment"
    )
    node_shapes: list[str] = Field(
        default=[],
        description="Node shapes of the SHACL shape to segment; empty for all"
    )
    html_example_path: str | None = Field(
        default=None,
        description="Example HTML document whose element ids match the shape's usage notes; none for no examples"
    )
    ontologies: list[str] = Field(
        default=[],
        description="Vocabularies whose rdfs:subClassOf hierarchy decides which shapes an existing entity can play"
    )
    usage_note_predicate: str | None = Field(
        default=None,
        description="Property of the shape's node and property shapes holding the id of their example element"
    )
    same_as_predicate: str | None = Field(
        default=None,
        description="Property whose (almost) equal value identifies the same entity in the triplestore, e.g. a title"
    )
    prompt_rules: list[str] = Field(
        default=[],
        description="Rules on how the target's documents are divided, added to the prompt's rules on copying text, "
                    "e.g. that an occurrence is never split at a heading"
    )
    numbered_parts: bool = Field(
        default=False,
        description="The occurrences of a structure are numbered, with numbered sub-parts (e.g. 'Artikel 1.1' under "
                    "'Artikel 1'): the prompt tells the LLM that a sub-part belongs to its part, and a sub-part or an "
                    "occurrence without a number after a numbered one that the LLM returns anyway is merged into the "
                    "occurrence before it"
    )
    same_as_class: str | None = Field(
        default=None,
        description="Segmented entities of this class are linked to the same entity in the triplestore (an original), "
                    "found by same_as_predicate; none to link nothing"
    )

    @model_validator(mode="after")
    def check_same_as(self):
        if self.same_as_class and not self.same_as_predicate:
            raise ValueError("same_as_class needs a same_as_predicate to find the same entity by")
        return self


class SegmentationConfig(BaseModel):
    """Segmentation model configuration for document structure extraction."""

    llm: LlmConfig = Field(
        default_factory=LlmConfig,
        description="LLM provider and model settings"
    )
    targets: dict[str, TargetConfig] = Field(
        min_length=1,
        description="Kinds of task inputs to segment, by name; an input is segmented with the first target (in this "
                    "order) whose target_filter matches it"
    )


class AppConfig(BaseSettings):
    """Root application configuration model."""
    
    model_config = SettingsConfigDict(
        extra="forbid", # Reject extra fields not defined in the model
        env_nested_delimiter="__",  # allows SEGMENTATION__LLM__API_KEY etc.
        env_ignore_empty=True,      # treat empty string env vars as unset
    )
    
    segmentation: SegmentationConfig = Field(
        default_factory=SegmentationConfig,
        description="Segmentation model configuration"
    )


def get_config() -> AppConfig:
    """
    Get the current configuration instance.
    
    If not yet loaded, attempts to load from default location.
    
    Returns:
        AppConfig instance
    """
    return load_config(AppConfig)
