# Planning

Planning records for `device-scheduler` belong here. Keep this directory concise during repository setup; the detailed design and implementation bootstrap will be added as a separate, authorized step.

## Workspace and authority

- Project Control workspace and authority repository: `device-scheduler`.
- Project ID: `5c878e94-2d97-4902-b594-9adb76c3e7fe`.
- Authority ID: `9b19daf8-d7c5-5e56-8b9f-df3630d94961`.
- Native metadata: `.todo-orchestrator/project.json`; canonical local state: `.todo-orchestrator/authority.sqlite3`.

The metadata is suitable for version control. The database and sidecars are ignored; credentials stay in Project Control's private host store. Do not hand-edit databases or generated projections, or initialize a replacement authority when opening another checkout.

For authorized workflow work, call `next_task` with project `device-scheduler`, then use its handle with `inspect_task`, `coordinate_task`, and `finish_task`. Use `overview` and `frontier` for read-only project and workflow context. Native Todo is the transactional authority behind these interfaces.

Setup verification on 2026-10-21: native schema 21, SQLite integrity `ok`, and observer, coder, mutator, investigator, and skill-assembler sessions verified. No implementation work, runs, or attempts were created. A comprehensive design and implementation plan remains a separate step.
