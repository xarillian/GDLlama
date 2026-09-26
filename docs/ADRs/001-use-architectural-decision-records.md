# ADR-001: Record durable architecture decisions as ADRs

Status: Accepted
Date: 2026-06-22

## Context

The original, uncommitted decision log mixed architectural commitments with roadmap changes, implementation notes, temporary planning constraints, and reversals. As it grew, durable reasoning became hard to find among entries whose value ended when their work landed or their plan changed.

Architecture decisions need a small, tracked record that explains their context and consequences after the implementation becomes familiar. Active planning decisions still need a local place where they can change without turning the architectural record into a progress ledger.

## Decision

Record each durable architecture decision in its own numbered file under `docs/ADRs/`. Use the compact title, status, and date header followed by Context, Decision, Consequences, Compliance, and Notes sections.

An ADR earns a place here when it constrains future architecture or preserves reasoning that code and `docs/ARCHITECTURE.md` do not explain on their own. Completed implementation choices, roadmap sequencing, status updates, and superseded planning details do not become ADRs. `_project/DECISIONS.md` holds only decisions that still bind unbuilt work.

Existing ADRs are immutable historical records except for status changes and corrections. A later decision that replaces one marks the old record as overwritten and points to its replacement.

## Consequences

Positive:

- Architectural reasoning remains visible after plans and implementation details change.
- One file per decision keeps review, linking, and later replacement focused.
- The tracked ADR set stays small enough to read as a coherent architectural history.

Negative:

- Contributors must judge whether a decision is durable enough to record.
- Some context remains split between the current architecture, an ADR, and active project planning.
- Replacing a decision requires maintaining links and status in more than one record.

## Compliance

- ADR filenames use a zero-padded sequential number and a short descriptive name.
- Each ADR includes the required header and sections defined by this record.
- Accepted architectural decisions live under `docs/ADRs/`, not in the local project ledger.
- Replaced ADRs remain present with an `Overwritten YYYY-MM-DD` status and a link to the replacing ADR.

## Notes

- Version: 1.01
- Changelog:
  - 1.01: Specified old decision log source.
  - 1.0: Initial accepted record
