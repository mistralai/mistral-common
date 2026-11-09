from typing import Any, TypeAlias

TimeoutType = float | int | str
r"""Accepted raw timeout values: any number of seconds, converted to float."""

JSONSchemaDict: TypeAlias = dict[str, Any] | None
r"""A JSON schema as a dictionary, or `None` when there is no schema."""
