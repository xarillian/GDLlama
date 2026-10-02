# Architecture Decision Records

Each record here explains one decision that binds future work on Chorus: its context, the decision, and its consequences. [ADR-001](001-use-architectural-decision-records.md) sets when a decision earns a record and the sections each one holds. Start new records from the [template](xxx-adr-template.md).

## Statuses

- **Proposed:** under review and not yet binding.
- **Accepted:** binding on new work.
- **Rejected:** considered and declined, kept for its reasoning.
- **Overwritten:** replaced by a later record, which it links to.

## Log

| ADR | Decision | Status | Date |
| --- | --- | --- | --- |
| [000](000-rewrite-gdllama-as-chorus.md) | Rewrite GDLlama as Chorus, a host- and provider-independent runtime | Proposed | 2026-07-01 |
| [001](001-use-architectural-decision-records.md) | Record durable architecture decisions as ADRs | Accepted | 2026-06-22 |
| [002](002-asynchronous-request-lifecycle.md) | Keep inference asynchronous and fence every request lifecycle | Accepted | 2026-07-11 |
| [003](003-domain-contract-dependencies.md) | Point dependencies toward domain contracts | Accepted | 2026-07-18 |
| [004](004-structured-logging.md) | Route structured log records to host-owned presentation | Accepted | 2026-08-16 |
| [005](005-typed-engine-signals.md) | Represent engine signals as typed variants | Accepted | 2026-08-30 |
| [006](006-express-public-generation-configuration-as-caller-intent.md) | Express public generation configuration as caller intent | Proposed | 2026-09-08 |
| [007](007-store-defaults-in-project-settings.md) | Persist generation defaults as portable JSON | Proposed | 2026-09-09 |
| [008](008-parallel-request-preparation.md) | Prepare requests in parallel and publish them in admission order | Proposed | 2026-09-30 |

Add a row with each new record, and update a row's status when its record changes.
