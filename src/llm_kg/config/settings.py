"""Secrets loaded from environment variables. Never written to YAML."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """All API-key-style secrets the framework knows about.

    Add new keys here as you add providers. Loaded from env vars or .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    wandb_api_key: SecretStr = SecretStr("")
