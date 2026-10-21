# Repository guidance

- Preserve the setup-only scope unless the user authorizes a design or implementation task.
- Keep package boundaries and the independence of the core described in `README.md`.
- Preserve unknown files, edits, history, active work, and Todo state. Do not reset or clean work to make the checkout appear empty.
- Delegate bounded engineering objectives to configured Luna profiles: `scout`, `researcher`, `implementer`, `debugger`, `optimizer`, `verifier`, `reviewer`, or `maintainer`. Give workers ownership and integration boundaries; accommodate concurrent changes. Do not use generic profiles or assign engineering to Project Control local workers. The root retains workflow lifecycle and final acceptance.
- `principal` uses Sol and requires an exceptional, evidence-backed allocation by the root. Only the user may authorize Astra. Workers must not self-promote or silently change models.
- Use supported Project Control tools for native Todo operations. Do not edit Todo databases or generated projections directly.
- For authorized workflow work, select workspace `device-scheduler` with `next_task`; use its handle with `inspect_task`, `coordinate_task`, and `finish_task`. Read-only context requests do not require a task claim.
- Verify affected contracts and report only checks that actually ran; state unavailable checks and evidence limits plainly.
