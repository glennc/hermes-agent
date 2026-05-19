# OpenTelemetry Observability Plugin

This bundled plugin is opt-in and emits Hermes turn, LLM, and tool spans with
OpenTelemetry. It is generic by default: any OTLP-compatible backend works with
standard `OTEL_EXPORTER_OTLP_*` environment variables.

## Enable

```bash
hermes plugins enable observability/otel
```

For a generic OTLP collector:

```bash
pip install 'hermes-agent[observability-otel]'
export OTEL_EXPORTER_OTLP_ENDPOINT=https://collector.example.com
```

For Azure Monitor/Application Insights:

```bash
pip install 'hermes-agent[observability-azure-monitor]'
export APPLICATIONINSIGHTS_CONNECTION_STRING='InstrumentationKey=...;IngestionEndpoint=...'
```

## Auto mode

With `HERMES_OTEL_EXPORTER=auto` or unset, the plugin selects:

1. `otlp` when `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set.
2. `azure_monitor` when `APPLICATIONINSIGHTS_CONNECTION_STRING` is set.
3. `console` only when `HERMES_OTEL_CONSOLE=true`.
4. disabled when none of the above is configured.

If both OTLP and Application Insights are present, OTLP wins so a user-supplied
collector endpoint is not overridden by Foundry's platform-injected connection
string.

## Foundry hosted agents

When Foundry environment variables are present, spans are decorated with Foundry
agent/session/project metadata. The Azure Monitor exporter is loaded only when
selected, and the plugin never emits Foundry's internal
`_msft_appinsights_connection` routing attribute.

## Optional settings

```bash
HERMES_OTEL_EXPORTER=auto              # auto, otlp, azure_monitor, console, disabled
HERMES_OTEL_CAPTURE_CONTENT=false      # opt-in prompt/tool/result content capture
HERMES_OTEL_MAX_ATTR_CHARS=2048        # truncation limit for captured content
HERMES_OTEL_DEBUG=false                # verbose plugin logging
HERMES_OTEL_SERVICE_NAME=hermes-agent  # OpenTelemetry service.name override
```

Content capture is off by default because prompts, completions, tool arguments,
and tool results can contain sensitive data.
