"""Environment-only configuration for the Hermes OpenTelemetry plugin."""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import Any

from .foundry import detect_foundry_attributes


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class OTelSettings:
    exporter: str
    enabled: bool
    reason: str = ""
    capture_content: bool = False
    max_attr_chars: int = 2048
    debug: bool = False
    service_name: str = "hermes-agent"
    service_namespace: str = "hermes"
    service_instance_id: str = ""
    resource_attributes: dict[str, Any] = field(default_factory=dict)
    span_attributes: dict[str, Any] = field(default_factory=dict)
    otlp_protocol: str = "http/protobuf"
    otlp_endpoint: str = ""
    otlp_traces_endpoint: str = ""
    azure_connection_string: str = ""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name).lower()
    if not value:
        return default
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _normalize_exporter(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    if value in {"", "auto"}:
        return "auto"
    if value in {"0", "false", "off", "none", "disabled", "disable"}:
        return "disabled"
    if value in {"azure", "azure_monitor", "appinsights", "application_insights"}:
        return "azure_monitor"
    if value in {"otlp", "generic_otlp"}:
        return "otlp"
    if value in {"console", "stdout"}:
        return "console"
    return value


def _normalize_otlp_protocol(value: str) -> str:
    value = value.strip().lower()
    if value in {"http", "http/protobuf", "http_proto", "protobuf"}:
        return "http/protobuf"
    if value in {"grpc", "http/grpc"}:
        return "grpc"
    return "http/protobuf"


def _parse_resource_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if not raw:
        return attrs
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            attrs[key] = value
    return attrs


def _service_identity(foundry_attrs: dict[str, Any]) -> tuple[str, str, str]:
    service_name = (
        _env("HERMES_OTEL_SERVICE_NAME")
        or str(foundry_attrs.get("microsoft.foundry.agent.name") or "")
        or "hermes-agent"
    )
    service_namespace = (
        _env("HERMES_OTEL_SERVICE_NAMESPACE")
        or ("foundry-hosted-agent" if foundry_attrs else "hermes")
    )
    service_instance_id = (
        _env("HERMES_OTEL_SERVICE_INSTANCE_ID")
        or str(foundry_attrs.get("microsoft.foundry.agent.session.id") or "")
        or socket.gethostname()
    )
    return service_name, service_namespace, service_instance_id


def resolve_settings() -> OTelSettings:
    """Resolve plugin settings from environment variables only.

    The plugin is opt-in through Hermes' plugin loader. Runtime activation is
    then driven by standard OpenTelemetry/Azure environment variables so
    non-Foundry users can enable it with only ``OTEL_EXPORTER_OTLP_ENDPOINT``.
    """
    foundry_attrs = detect_foundry_attributes()
    service_name, service_namespace, service_instance_id = _service_identity(foundry_attrs)
    resource_attrs: dict[str, Any] = {
        "service.name": service_name,
        "service.namespace": service_namespace,
        "service.instance.id": service_instance_id,
        "telemetry.sdk.name": "opentelemetry",
        "hermes.observability.plugin": "otel",
    }
    resource_attrs.update(_parse_resource_attributes(_env("OTEL_RESOURCE_ATTRIBUTES")))
    resource_attrs.update(_parse_resource_attributes(_env("HERMES_OTEL_RESOURCE_ATTRIBUTES")))

    otlp_endpoint = _env("HERMES_OTEL_EXPORTER_OTLP_ENDPOINT") or _env("OTEL_EXPORTER_OTLP_ENDPOINT")
    otlp_traces_endpoint = (
        _env("HERMES_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or _env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )
    otlp_protocol = _normalize_otlp_protocol(
        _env("HERMES_OTEL_EXPORTER_OTLP_PROTOCOL")
        or _env("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or _env("OTEL_EXPORTER_OTLP_PROTOCOL")
        or "http/protobuf"
    )
    azure_connection_string = _env("APPLICATIONINSIGHTS_CONNECTION_STRING")

    explicit = _normalize_exporter(_env("HERMES_OTEL_EXPORTER", "auto"))
    exporter = explicit
    reason = ""

    if explicit == "auto":
        if otlp_endpoint or otlp_traces_endpoint:
            exporter = "otlp"
        elif azure_connection_string:
            exporter = "azure_monitor"
        elif _env_bool("HERMES_OTEL_CONSOLE", False):
            exporter = "console"
        else:
            return OTelSettings(
                exporter="disabled",
                enabled=False,
                reason="no OTLP endpoint or Application Insights connection string",
                capture_content=_env_bool("HERMES_OTEL_CAPTURE_CONTENT", False),
                max_attr_chars=_env_int("HERMES_OTEL_MAX_ATTR_CHARS", 2048),
                debug=_env_bool("HERMES_OTEL_DEBUG", False),
                service_name=service_name,
                service_namespace=service_namespace,
                service_instance_id=service_instance_id,
                resource_attributes=resource_attrs,
                span_attributes=foundry_attrs,
                otlp_protocol=otlp_protocol,
                otlp_endpoint=otlp_endpoint,
                otlp_traces_endpoint=otlp_traces_endpoint,
                azure_connection_string=azure_connection_string,
            )

    if exporter == "azure_monitor" and not azure_connection_string:
        return OTelSettings(
            exporter=exporter,
            enabled=False,
            reason="APPLICATIONINSIGHTS_CONNECTION_STRING is empty",
            capture_content=_env_bool("HERMES_OTEL_CAPTURE_CONTENT", False),
            max_attr_chars=_env_int("HERMES_OTEL_MAX_ATTR_CHARS", 2048),
            debug=_env_bool("HERMES_OTEL_DEBUG", False),
            service_name=service_name,
            service_namespace=service_namespace,
            service_instance_id=service_instance_id,
            resource_attributes=resource_attrs,
            span_attributes=foundry_attrs,
            otlp_protocol=otlp_protocol,
            otlp_endpoint=otlp_endpoint,
            otlp_traces_endpoint=otlp_traces_endpoint,
            azure_connection_string=azure_connection_string,
        )

    if exporter not in {"otlp", "azure_monitor", "console"}:
        return OTelSettings(
            exporter=exporter,
            enabled=False,
            reason=f"unsupported exporter {exporter!r}",
            capture_content=_env_bool("HERMES_OTEL_CAPTURE_CONTENT", False),
            max_attr_chars=_env_int("HERMES_OTEL_MAX_ATTR_CHARS", 2048),
            debug=_env_bool("HERMES_OTEL_DEBUG", False),
            service_name=service_name,
            service_namespace=service_namespace,
            service_instance_id=service_instance_id,
            resource_attributes=resource_attrs,
            span_attributes=foundry_attrs,
            otlp_protocol=otlp_protocol,
            otlp_endpoint=otlp_endpoint,
            otlp_traces_endpoint=otlp_traces_endpoint,
            azure_connection_string=azure_connection_string,
        )

    return OTelSettings(
        exporter=exporter,
        enabled=True,
        reason=reason,
        capture_content=_env_bool("HERMES_OTEL_CAPTURE_CONTENT", False),
        max_attr_chars=_env_int("HERMES_OTEL_MAX_ATTR_CHARS", 2048),
        debug=_env_bool("HERMES_OTEL_DEBUG", False),
        service_name=service_name,
        service_namespace=service_namespace,
        service_instance_id=service_instance_id,
        resource_attributes=resource_attrs,
        span_attributes=foundry_attrs,
        otlp_protocol=otlp_protocol,
        otlp_endpoint=otlp_endpoint,
        otlp_traces_endpoint=otlp_traces_endpoint,
        azure_connection_string=azure_connection_string,
    )
