# Repository Instructions

- Work only inside this repository.
- Never read, print, edit, stage, commit, or summarize secret values.
- Treat these files as sensitive and off limits: `config/secrets.env`, `secrets/development.env`, `.env.dryrun`, `.env`, and `*.env`.
- Before any major work, inspect the repository and produce a phased plan.
- For large overhauls, split work into small, reversible phases.
- Prefer architecture cleanup, deduplication, dependency cleanup, test reliability, documentation accuracy, and removing dead code.
- Do not rewrite large parts of the codebase without first explaining the risk.
- Do not delete files unless they are clearly generated, duplicated, obsolete, or explicitly approved.
- After edits, run relevant tests or explain why tests could not run.
- Before any commit, run `git status` and confirm no secret files are tracked.
- Do not commit or push unless explicitly asked.
- When uncertain, ask before destructive changes.
