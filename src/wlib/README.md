# wLib

Module agnostic utilities. Truly generic and extremely reusable tools that are at home in any project.

Rules:
- Depends on the standard library only. No external or internal libs.
- Any layer in the project can include it.
- Headers live here (private) until a downstream consumer earns them a spot.
- Can include tools that are useful as one-offs and are unsued in code.
