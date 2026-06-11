# Codex Workflow

## Required Orientation

For future repository tasks:

1. Read `AGENTS.md`.
2. Read `docs/NEWS_PIPELINE_CONTEXT.md`.
3. Inspect the current repository state and use it as the source of truth.
4. Apply only the requested delta.
5. Keep the final summary short unless a full report is requested.
6. At the end of every major phase, update
   `docs/NEWS_PIPELINE_CONTEXT.md` with the current state, known issues, and
   next recommended phase.

Do not assume a prior conversation accurately represents the current
worktree. Preserve unrelated user changes and do not revert them.

## Standard Working Method

1. Confirm the task constraints and prohibited actions.
2. Inspect only the relevant source, tests, and documentation.
3. Produce a phased plan before major work.
4. Keep changes small, reversible, and aligned with existing patterns.
5. Add or update tests proportional to the behavioral risk.
6. Run the requested validation or explain why it could not run.
7. Audit the changed-file list and generated outputs.
8. Do not commit or push unless explicitly requested.

Secret and environment files remain off limits. CLI environment diagnostics
may report only `set` or `missing`; they must never expose values.

## Short Prompt Template

```text
Read AGENTS.md and docs/NEWS_PIPELINE_CONTEXT.md.

Task: <one sentence>

Constraints: Use the standard project constraints from the context doc.

Implement:
1. <requested change>
2. <requested change>

Tests:
- <specific tests>

Validation:
- <specific commands>

Return:
1. Changed files
2. Tests and result
3. Key metrics
4. Remaining issues
5. Commit recommendation

Do not commit or push.
```

## Prompt Guidance

- State only constraints that differ from the permanent context.
- Name exact files or behaviors when the scope must remain narrow.
- Explicitly distinguish implementation work from investigation, review, or
  planning-only work.
- Include live validation commands only when network use is intended.
- State whether a send command is preview-only. Never imply permission to use
  `--confirm-send`.
- Request exact final formatting only when downstream parsing requires it.

## End-Of-Phase Context Update

When a major phase finishes, update `docs/NEWS_PIPELINE_CONTEXT.md` in the same
change:

- Replace stale architecture statements.
- Record newly implemented and validated capabilities.
- Remove roadmap items that are complete.
- Preserve unresolved reliability, quota, concentration, and data-quality
  issues.
- Name the next recommended phase without implementing it unless requested.

Do not include secret values, transient artifact contents, or verbose run logs
in permanent context.
