# Independent two-agent collaboration — 3D Terrain–Foothold World Model

This package gives ChatGPT Work a **concrete local setup job**. It **does not automatically connect this chat** to your Windows machine or SSH host. Extract it through ChatGPT Desktop Work/Computer Use, or copy it manually, then run the orchestrator in the terminal **on your RTX 4080 host**.

## Included

- `terrain_foothold_world_model.py`: current single-file research starter (synthetic functional tests previously exercised; **not yet trained/evaluated on real data**). No old weights. Source intentionally kept inspectable and independently audited; some contractual requirements may still be missing.
- `ARCHITECTURE_CONTRACT.md`: non-negotiable requirements, data gates, tests.
- `two_agent_collab.py`: actual Python orchestration: **Codex implementer ↔ independent Claude reviewer**, passing each other's report between rounds. Optional second Codex reviewer if Claude unavailable. No paid API calls by this script itself; the separately installed CLIs may incur usage.
- `WORK_HANDOFF.md`: precise instruction to paste into local ChatGPT Desktop **Work**, which can request permission for local files and apps.

## Recommended procedure

1. On Windows desktop ChatGPT, choose **ChatGPT → Work**, open local/Computer Use as offered and give it `WORK_HANDOFF.md` and this ZIP. Permissions will need to be granted there.
2. Work copies this package into a new folder **on your remote SSH Linux host** (don't merge with the old project).
3. On that server, install/authenticate two CLIs according to [Codex CLI documentation](https://learn.chatgpt.com/docs/cli/reference) and [Claude Code CLI documentation](https://code.claude.com/docs/en/cli-reference), with your approval. Having a ChatGPT Plus subscription doesn't itself guarantee Claude Code access or free API usage.
4. Work creates a dedicated Git baseline, then runs:

```bash
python two_agent_collab.py --check
python two_agent_collab.py --run --rounds 2 --reviewer claude --smoke "python terrain_foothold_world_model.py self-test"
```

If Claude is not available but Codex is, an optional fallback uses two isolated Codex CLI sessions:

```bash
python two_agent_collab.py --run --rounds 2 --reviewer codex --smoke "python terrain_foothold_world_model.py self-test"
```

The orchestrator refuses to run if either CLI is missing, a tracked baseline isn't clean, or the folder lacks required architecture files. It limits rounds to 1–5 and process timeouts; it never self-schedules background activity. The CLI process can read source and produce edits; inspect provider access and terms before approving its use.

## What this setup does NOT claim

- It does not remote-control your PC by itself, authorize SSH, install the CLIs, provide Claude authentication, install CUDA, or start real GPU training.
- It does not turn synthetic execution tests into proof of 3D physical accuracy. Reconstructed surfels are not omniscient true geometry, and point tracking remains uncertain.
- It does not guarantee the starting model fully satisfies the contract. The point of independent review is to surface and fix remaining mismatches before real training.
- It cannot produce contact-force supervision from data lacking a reliable force stream.

## Safety and privacy

Do not paste credentials in a Work conversation, Git repository or prompt. Source code may be sent to each model provider when you run its CLI. Approve access and possible billing before launching. Do not use model output directly as a robot controller without independent safety engineering and validation.
