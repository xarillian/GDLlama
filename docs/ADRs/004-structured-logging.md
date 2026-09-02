# ADR-004: Route structured log records to host-owned presentation

Status: Accepted
Date: 2026-08-16

## Context

The original logging path called host code directly from provider threads. That violated the same thread and lifetime boundary used for inference events, and the process-global llama.cpp hook could not safely belong to a host callback. Core-side formatting and stderr output also imposed presentation policy on every host.

Earlier designs added a source string to every record and used a severity bitmask. No host consumed source filtering, and arbitrary severity subsets could represent incoherent choices such as showing informational messages while hiding warnings.

## Decision

`Logger` creates structured `LogRecord` values, applies an ordered minimum severity threshold, adds available request and session context, and pushes records to a shared `LogChannel`. Hosts drain records on their own thread and own all formatting, display, persistence, and crash-reporting policy.

Log records carry no provider or vendor source field. Logging has its own channel and ordering relative to inference events is not guaranteed. Logging callbacks are observational and never control inference flow.

## Consequences

Positive:

- Provider threads never invoke host presentation code.
- Every host can present the same structured records in its own idiom.
- An ordered threshold has one clear meaning and avoids a second severity vocabulary.
- Records do not pay for unused source attribution.

Negative:

- Hosts must poll and present log records themselves.
- Log and inference event order cannot be reconstructed across their separate channels.
- Process-global vendor output cannot be attributed reliably when several engines are live.
- Crash artifacts require explicit host support.

## Compliance

- Providers receive a `Logger`; they do not receive host callbacks or write presentation output.
- Hosts drain `LogChannel` records on the host thread.
- Severity filtering uses a minimum `LogLevel`, including `LogLevel::Off`, rather than a mask.
- Core and provider code must not format logs for a particular host.

## Notes

- Version: 1.0
- Changelog:
  - 1.0: Initial accepted record combining the final logging design and its superseded drafts
