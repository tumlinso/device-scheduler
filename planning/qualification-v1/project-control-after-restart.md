# Project Control access after the operator restart

Observed 2026-10-21, approximately 15:12–15:31 UTC.

- The live MCP frontdoor resolves workspace `device-scheduler` and its registered
  repository. Native authority `9b19daf8-d7c5-5e56-8b9f-df3630d94961`, project
  `5c878e94-2d97-4902-b594-9adb76c3e7fe`, remained readable at revision 26,
  including all 15 work definitions/assignments, the accepted plan, and DS-00's
  preserved attempt.
- The overview's semantic summary was revision 7, while native context was
  revision 26. Overview counts are therefore insufficient to infer task/run state.
- `next_task` returned `workflow_recovery_required` for the existing attempt.
  `inspect_task` could read task context and recovery state. DS-00's lease had
  expired at 14:43:01 UTC; source was clean, children absent, registered producer
  liveness unknown, and no recovery intents or outcome candidates were present.
- Retrying the pending `finish_task(complete)` returned
  `native_operation_rejected`; no task completion or acceptance is claimed.
- A host-authorized read through `project-control todo call ...
  todo_inspect_attempt` reproduced the source-clean recovery observation.
  Sandboxed direct CLI observations could not establish source safety and must
  not be mistaken for dirty source evidence.
- The live user service was PID 128745. Reading only its resource-path
  environment variables showed
  `TODO_BACKGROUND_HOST_RUNTIME_DIR=/tmp/codex-todo-orchestrator-1000` and
  `XDG_RUNTIME_DIR=/run/user/1000`.
- Host-authorized path checks found no `physical-resources.sqlite3` at that
  configured path, the XDG fallback, or the configured path through the service's
  `/proc/128745/root` mount namespace. The native handback provider requires an
  available physical-owner query. The missing configured registry is an
  operational blocker; the generic finish rejection alone does not prove that
  this is its only cause.

No authority database or generated projection was edited. No substitute physical
ownership database was initialized. The native DS-00 task workspace and its
committed baseline remain preserved. Recover the configured ownership source
through Project Control's supported operational procedure before accepting
native completion or transferring resource authority. Implementation and local
qualification continue under the user's explicit implementation mandate.

## Follow-up: supported registry creation and expired-attempt release

The following follow-up was performed through Project Control's supported
`HostCoordinator` and native workflow routes under the user's explicit rule
exception. It supersedes the earlier blocker for current registry access, but
does not reconstruct prior physical ownership history or establish task
acceptance.

- Before registry creation, fresh host queries showed zero GPU processes and
  zero GPU services. The supported coordinator created the previously missing
  registry. Its resource count is zero; historical ownership was not recovered.
  This establishes current registry availability only, not a historical
  ownership transfer or GPU qualification.
- The expired DS-00 attempt was released through the supported, versioned
  native recovery intent after public `next_task` remained stuck. Native receipt
  `623ed0b2c9d943338c6198049c165b21` records revision 27. After that native
  release, a backed-up private compact-pointer repair atomically cleared stale
  `handback_pending` state. No Todo database or generated projection was edited
  directly.
- The new DS-00 source `6ce074b` was copied from the preserved baseline
  `eeed0eb`. It was reviewed and rejected because the authored check facts were
  missing; there is no pending acceptance. This rejection was the reason the
  native plan verification requirements are now being repaired.
- A fresh MCP read at 16:24:57 UTC reported native revision 36 and overview
  snapshot 7. The maintained systemd user service was active/running as PID
  128745 with `ExecMainStatus=0`.

The earlier revision-26 recovery observations remain useful chronology: they
record the state before the supported registry creation and native attempt
release. Current qualification must continue to distinguish the empty newly
created registry from any historical allocator/lease ownership, which remains
unrecovered. The current source baseline has not been accepted as DS-00 and no
implementation completion is claimed by this operational update.
