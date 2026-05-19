"""OTLP exporter wiring for the OpenTelemetry plugin."""
from __future__ import annotations

from typing import Any


def _ensure_otlp_deps():
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        return BatchSpanProcessor
    except ImportError:
        from tools.lazy_deps import ensure

        ensure("observability.otel", prompt=False)
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        return BatchSpanProcessor


def _http_endpoint(settings: Any) -> str | None:
    if settings.otlp_traces_endpoint:
        return settings.otlp_traces_endpoint
    if not settings.otlp_endpoint:
        return None
    endpoint = settings.otlp_endpoint.rstrip("/")
    if endpoint.endswith("/v1/traces"):
        return endpoint
    return f"{endpoint}/v1/traces"


def _grpc_endpoint(settings: Any) -> str | None:
    return settings.otlp_traces_endpoint or settings.otlp_endpoint or None


def build_span_processor(settings: Any):
    BatchSpanProcessor = _ensure_otlp_deps()
    if settings.otlp_protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        endpoint = _grpc_endpoint(settings)
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = _http_endpoint(settings)

    kwargs = {"endpoint": endpoint} if endpoint else {}
    return BatchSpanProcessor(OTLPSpanExporter(**kwargs))
