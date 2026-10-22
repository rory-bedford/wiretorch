"""The base class of every configuration section current code reads."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, model_validator


class Config(BaseModel):
    """A configuration section in which an unknown key is an error.

    A misspelt key would otherwise be dropped silently, leaving its default in force.
    Keys a section no longer has, but which existing files still carry, are listed in
    ``LEGACY_KEYS``: accepted and ignored.
    """

    model_config = ConfigDict(extra="forbid")

    LEGACY_KEYS: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_keys(cls, data):
        if isinstance(data, dict) and cls.LEGACY_KEYS:
            data = {k: v for k, v in data.items() if k not in cls.LEGACY_KEYS}
        return data
