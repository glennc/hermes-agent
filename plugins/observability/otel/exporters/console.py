"""Console exporter wiring for the OpenTelemetry plugin."""
from __future__ import annotations


def _ensure_console_deps():
    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        return ConsoleSpanExporter, SimpleSpanProcessor
    except ImportError:
        from tools.lazy_deps import ensure

        ensure("observability.otel", prompt=False)
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        return ConsoleSpanExporter, SimpleSpanProcessor


def build_span_processor(_settings):
    ConsoleSpanExporter, SimpleSpanProcessor = _ensure_console_deps()
    return SimpleSpanProcessor(ConsoleSpanExporter())

