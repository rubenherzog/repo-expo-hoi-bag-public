# Repository contract for agents

Read this file and [docs/REPOSITORY_CONTRACT.md](docs/REPOSITORY_CONTRACT.md) in full before inspecting, editing, running, or creating anything. These are binding repository rules, not suggestions.

## Authority and scope

- The current checkout is self-contained and definitive. Do not make it depend at runtime on a sibling, earlier, private, or source repository.
- A source repository may be consulted read-only only when the user explicitly requests a migration or comparison. Port the resulting contract, configuration, code, and tests into this checkout; do not leave file links, imports, paths, or runtime dependencies to the source checkout.
- User instructions override this document. Preserve unrelated user changes in a dirty worktree.
- Treat versioned public inputs, delivered references, and `outputs/` as read-only unless the user explicitly requests a change. Never delete, rename, move, or overwrite them by inference.

## Required preflight

1. Read `README.md`, `docs/REPOSITORY_CONTRACT.md`, and the relevant files in `config/`.
2. For a stage or runtime task, inspect its entry point, `legacy_runtime.py` when applicable, and the existing output contract before choosing any path or command.
3. For a source migration, compare the exact source files and record the local destination. Do not reconstruct a contract from memory.
4. Before a write, state the exact target and why it is inside the repository contract. Never create convenience directories, duplicate checkouts, symlinks, or ad-hoc runtime roots.

## Runtime and outputs

- The checkout contains source, configuration, public inputs, documentation, tests, and delivered references. Heavy generated results, checkpoints, logs, matrices, models, and temporary runtime files belong under the operator-selected external `REPRO_DATA_ROOT`.
- Production commands must pass that external runtime explicitly as `--repro-data-root` (or export it as `REPRO_DATA_ROOT`). Do not substitute a sibling checkout or a prior analysis runtime.
- Do not create repository-root links to external results. Do not write generated artifacts to a parent directory, a sibling directory, or an invented output tree.
- `outputs/` is an existing local delivery area. Keep its structure intact and never delete its contents. Follow the exact target defined by the invoked stage and configuration.
- A long run must use the repository's established launcher/runtime path, persist resumable state only in its configured runtime location, and emit useful progress to its configured log destination. Do not silently suppress warnings: diagnose or handle the relevant condition and preserve actionable diagnostics.

## Configuration and scientific invariants

- `config/paper.yaml` holds public analysis parameters. `config/country_exclusions.yaml` holds the independent greedy and BAG population policies; read [docs/COUNTRY_EXCLUSION_CONTRACT.md](docs/COUNTRY_EXCLUSION_CONTRACT.md) before changing a stage that filters countries or diagnoses.
- Preserve primary-BAG order: structural, then functional. Combined BAG is opt-in/supplementary.
- New stages must consume local configuration, carry relevant configuration/input hashes into resumable artifacts, and add tests for their contract.
- Legacy compatibility code may use `V3_*` variables only at the compatibility boundary. New package code should use local configuration and `RunContext`.

## Verification and handoff

- Use read-only inspection first. After a code/configuration change, run the smallest relevant tests, then broader tests when the change crosses shared infrastructure.
- Report what changed, what was verified, the exact artifact location if one was intentionally produced, and any known limitation. Never claim a run succeeded without reading its final status or manifest.
