"""
Configuration Management with Pydantic Validation

This module provides Pydantic models for validating application configuration
loaded from config.json.
"""

from pydantic import BaseModel, Field, SecretStr
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


class ShapeConfig(BaseModel):
    """Input of the shape-driven segmentor (LLMSegmentorWithShape). Relative paths are resolved against the project root."""

    shape_path: str = Field(
        default="shapes/notulen-shape.ttl",
        description="SHACL shape (Turtle) describing what to segment"
    )
    node_shapes: list[str] = Field(
        default=[
            "https://data.vlaanderen.be/shacl/besluit-publicatie#BesluitShape",
            "https://data.vlaanderen.be/shacl/besluit-publicatie#ArtikelShape",
        ],
        description="Node shapes of the SHACL shape to segment; empty for all"
    )
    example_paths: list[str] = Field(
        default=["shapes/notulen-example.html"],
        description="Example HTML documents whose element ids match the shape's usage notes"
    )
    ontologies: list[str] = Field(
        default=["https://data.vlaanderen.be/ns/besluit", "http://data.europa.eu/eli/ontology"],
        description="Vocabularies whose rdfs:subClassOf hierarchy decides which shapes an existing entity can play"
    )
    usage_note_predicate: str = Field(
        default="http://lblod.data.gift/vocabularies/besluit/usageNote",
        description="Property of the shape's node and property shapes holding the id of their example element"
    )
    text_predicate: str = Field(
        default="https://data.europarl.europa.eu/def/epvoc#expressionContent",
        description="Property of an input resource holding the text to segment"
    )
    same_as_predicate: str = Field(
        default="http://data.europa.eu/eli/ontology#title",
        description="Property whose (almost) equal value identifies the same entity in the triplestore, e.g. a title"
    )
    same_as_class: str | None = Field(
        default="http://data.vlaanderen.be/ns/besluit#Besluit",
        description="Segmented entities of this class are linked to the same entity in the triplestore (an original); "
                    "none to link nothing"
    )


class SegmentationConfig(BaseModel):
    """Segmentation model configuration for document structure extraction."""

    llm: LlmConfig = Field(
        default_factory=LlmConfig,
        description="LLM provider and model settings"
    )
    shape: ShapeConfig = Field(
        default_factory=ShapeConfig,
        description="Shape, examples and vocabularies for the shape-driven segmentor"
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
