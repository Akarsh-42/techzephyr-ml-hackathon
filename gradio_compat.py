"""Project-local compatibility guard for Gradio 4.44 / gradio-client 1.3.

Pydantic 2.11 emits boolean JSON Schema values for unconstrained dictionary
fields. gradio-client 1.3's API-info parser assumes nested schemas are dicts.
Keep the workaround in project code so deployments do not need patched files
under site-packages.
"""
from __future__ import annotations

import gradio
import gradio_client
from gradio_client import utils as client_utils


def install_json_schema_compat() -> bool:
    """Guard the known vulnerable dependency pair; return whether installed."""
    if gradio.__version__ != "4.44.0" or gradio_client.__version__ != "1.3.0":
        return False

    if getattr(client_utils, "_techzephyr_json_schema_compat", False):
        return True

    original = client_utils._json_schema_to_python_type

    def json_schema_to_python_type_compat(schema, defs):
        # JSON Schema permits booleans as schemas (true = unconstrained, false
        # = disallowed). The old parser treats nested boolean schemas as Any.
        if not isinstance(schema, dict) or schema == {}:
            return "Any"

        # gradio-client 1.3 recursively parses additionalProperties without
        # accounting for Pydantic's boolean shorthand. Ignore that constraint
        # when producing its descriptive Python type string.
        if "additionalProperties" in schema and isinstance(
            schema["additionalProperties"], bool
        ):
            schema = {key: value for key, value in schema.items() if key != "additionalProperties"}

        return original(schema, defs)

    client_utils._json_schema_to_python_type = json_schema_to_python_type_compat
    client_utils._techzephyr_json_schema_compat = True
    return True


install_json_schema_compat()
