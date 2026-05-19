"""Foundry-hosted agent environment detection for OpenTelemetry spans."""
from __future__ import annotations

import os
from typing import Any


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def detect_foundry_attributes() -> dict[str, Any]:
    """Return Foundry attributes when Hermes is running in a hosted agent.

    This function intentionally reads only public/platform-injected
    environment variables. It does not query Azure and never emits the
    internal ``_msft_appinsights_connection`` routing attribute.
    """
    hosting_environment = _env("FOUNDRY_HOSTING_ENVIRONMENT")
    project_endpoint = _env("FOUNDRY_PROJECT_ENDPOINT")
    project_arm_id = _env("FOUNDRY_PROJECT_ARM_ID")
    agent_name = _env("FOUNDRY_AGENT_NAME")
    agent_version = _env("FOUNDRY_AGENT_VERSION")
    agent_session_id = _env("FOUNDRY_AGENT_SESSION_ID")

    if not any(
        (
            hosting_environment,
            project_endpoint,
            project_arm_id,
            agent_name,
            agent_session_id,
        )
    ):
        return {}

    attrs: dict[str, Any] = {
        "microsoft.foundry.detected": True,
    }
    if hosting_environment:
        attrs["microsoft.foundry.hosting_environment"] = hosting_environment
    if project_endpoint:
        attrs["microsoft.foundry.project.endpoint"] = project_endpoint
    if project_arm_id:
        attrs["microsoft.foundry.project.id"] = project_arm_id
        attrs["gen_ai.azure_ai_project.id"] = project_arm_id
    if agent_name:
        attrs["microsoft.foundry.agent.name"] = agent_name
        attrs["gen_ai.agent.name"] = agent_name
    if agent_version:
        attrs["microsoft.foundry.agent.version"] = agent_version
    if agent_session_id:
        attrs["microsoft.foundry.agent.session.id"] = agent_session_id
        attrs["gen_ai.conversation.id"] = agent_session_id
    return attrs
