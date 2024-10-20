from .base import EnvConfig

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaworldConfig(EnvConfig):
    reward_func_version: str | None = None
