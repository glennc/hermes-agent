"""OpenTelemetry SDK runtime wiring for the Hermes OTel plugin."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import OTelSettings

logger = logging.getLogger(__name__)


@dataclass
class OTelBackend:
    tracer: Any
    provider: Any
    settings: OTelSettings

    def start_span(self, name: str, *, attributes: dict[str, Any], parent: Any = None) -> Any:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind

        context = trace.set_span_in_context(parent) if parent is not None else None
        return self.tracer.start_span(
            name,
            context=context,
            kind=SpanKind.INTERNAL,
            attributes=_clean_attributes(attributes),
        )

    def end_span(
        self,
        span: Any,
        *,
        attributes: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if span is None:
            return
        try:
            if attributes:
                for key, value in _clean_attributes(attributes).items():
                    span.set_attribute(key, value)
            if error:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, error))
            span.end()
        except Exception as exc:  # pragma: no cover - fail-open guard
            logger.debug("OTel span end failed: %s", exc)

    def flush(self) -> None:
        try:
            self.provider.force_flush()
        except Exception as exc:  # pragma: no cover - fail-open guard
            logger.debug("OTel force_flush failed: %s", exc)

    def shutdown(self) -> None:
        try:
            self.provider.shutdown()
        except Exception as exc:  # pragma: no cover - fail-open guard
            logger.debug("OTel shutdown failed: %s", exc)


def _ensure_sdk():
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
        from opentelemetry.trace import ProxyTracerProvider
        return trace, Resource, SDKTracerProvider, ProxyTracerProvider
    except ImportError:
        from tools.lazy_deps import ensure

        ensure("observability.otel", prompt=False)
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
        from opentelemetry.trace import ProxyTracerProvider
        return trace, Resource, SDKTracerProvider, ProxyTracerProvider


def _build_processor(settings: OTelSettings):
    if settings.exporter == "otlp":
        from .exporters.otlp import build_span_processor
    elif settings.exporter == "azure_monitor":
        from .exporters.azure_monitor import build_span_processor
    elif settings.exporter == "console":
        from .exporters.console import build_span_processor
    else:
        raise RuntimeError(f"Unsupported OTel exporter {settings.exporter!r}")
    return build_span_processor(settings)


def _clean_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            clean[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, bool, int, float)) for item in value
        ):
            clean[key] = list(value)
        else:
            clean[key] = str(value)
    return clean


def build_backend(settings: OTelSettings) -> OTelBackend:
    trace, Resource, SDKTracerProvider, ProxyTracerProvider = _ensure_sdk()
    current_provider = trace.get_tracer_provider()
    span_processor = _build_processor(settings)

    if isinstance(current_provider, ProxyTracerProvider):
        provider = SDKTracerProvider(
            resource=Resource.create(_clean_attributes(settings.resource_attributes))
        )
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)
    elif isinstance(current_provider, SDKTracerProvider):
        provider = current_provider
        provider.add_span_processor(span_processor)
    else:
        raise RuntimeError(
            f"OpenTelemetry tracer provider {type(current_provider).__name__} "
            "is not supported by the Hermes OTel plugin"
        )

    tracer = trace.get_tracer("hermes.agent", "0.1.0")
    return OTelBackend(tracer=tracer, provider=provider, settings=settings)
