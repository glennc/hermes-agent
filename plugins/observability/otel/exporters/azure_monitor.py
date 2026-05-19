"""Azure Monitor exporter wiring for the OpenTelemetry plugin."""
from __future__ import annotations

from typing import Any


def _ensure_azure_deps():
    try:
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        return AzureMonitorTraceExporter, BatchSpanProcessor
    except ImportError:
        from tools.lazy_deps import ensure

        ensure("observability.azure_monitor", prompt=False)
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        return AzureMonitorTraceExporter, BatchSpanProcessor


def build_span_processor(settings: Any):
    AzureMonitorTraceExporter, BatchSpanProcessor = _ensure_azure_deps()
    return BatchSpanProcessor(
        AzureMonitorTraceExporter(connection_string=settings.azure_connection_string)
    )

