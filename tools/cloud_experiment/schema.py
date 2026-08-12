"""JSON schema for the public cloud experiment manifest."""

from __future__ import annotations

from typing import Any


COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["argv", "timeout_seconds"],
    "properties": {
        "argv": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "timeout_seconds": {"type": "integer", "minimum": 1},
        "environment": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "required_outputs": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "checkpoint_paths": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "checkpoint_on_failure": {"type": "boolean"},
    },
}

PHASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "argv", "timeout_seconds"],
    "properties": {
        "id": {
            "type": "string",
            "pattern": "^[a-z0-9][a-z0-9-]{0,63}$",
        },
        **COMMAND_SCHEMA["properties"],
    },
}


MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://sair.local/schemas/cloud-experiment-v1.json",
    "title": "SAIR experiment-agnostic cloud lifecycle manifest",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "experiment",
        "package",
        "execution",
        "provider",
        "budget",
        "artifacts",
    ],
    "properties": {
        "schema_version": {"const": "sair-cloud-experiment-v1"},
        "experiment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "description"],
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": "^[a-z0-9][a-z0-9-]{2,79}$",
                },
                "description": {"type": "string", "minLength": 1},
            },
        },
        "package": {
            "type": "object",
            "additionalProperties": False,
            "required": ["include", "exclude", "require_clean"],
            "properties": {
                "include": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "exclude": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "require_clean": {"type": "boolean"},
            },
        },
        "execution": {
            "type": "object",
            "additionalProperties": False,
            "required": ["working_directory", "canary", "phases"],
            "properties": {
                "working_directory": {"type": "string", "minLength": 1},
                "environment": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "canary": COMMAND_SCHEMA,
                "phases": {
                    "type": "array",
                    "minItems": 1,
                    "items": PHASE_SCHEMA,
                },
            },
        },
        "provider": {
            "type": "object",
            "additionalProperties": False,
            "required": ["driver", "paid", "options", "expected_hardware"],
            "properties": {
                "driver": {
                    "type": "string",
                    "pattern": "^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$",
                },
                "paid": {"type": "boolean"},
                "options": {"type": "object"},
                "expected_hardware": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["accelerator", "accelerator_count"],
                    "properties": {
                        "accelerator": {"type": "string", "minLength": 1},
                        "accelerator_count": {"type": "integer", "minimum": 0},
                    },
                },
            },
        },
        "budget": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "max_total_usd",
                "max_provider_seconds",
                "cleanup_reserve_seconds",
                "max_idle_seconds",
            ],
            "properties": {
                "max_total_usd": {"type": "number", "exclusiveMinimum": 0},
                "max_provider_seconds": {"type": "integer", "minimum": 1},
                "cleanup_reserve_seconds": {"type": "integer", "minimum": 1},
                "max_idle_seconds": {"type": "integer", "minimum": 1},
            },
        },
        "artifacts": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_directory", "publication_directory"],
            "properties": {
                "run_directory": {"type": "string", "minLength": 1},
                "publication_directory": {"type": "string", "minLength": 1},
            },
        },
    },
}
