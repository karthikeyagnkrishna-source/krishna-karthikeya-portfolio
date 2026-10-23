# Independent two-agent collaboration — 3D Terrain–Foothold World Model

This package gives ChatGPT Work a **concrete local setup job**. It **does not automatically connect this chat** to your Windows machine or SSH host. Extract it through ChatGPT Desktop Work/Computer Use, or copy it manually, then run the orchestrator in the terminal **on your RTX 4080 host**.

## Included

- `terrain_foothold_world_model.py`: unchanged single-file research starter. A bounded RTX 4080 pilot on logged TartanGround simulator data was run on 10 October 2026; this is not proof of accuracy or deployment readiness. No old weights. Some contractual requirements remain missing.
- `prepare_tartanground_recordings.py`: separate, traceable data converter for four inspected TartanGround trajectories. It synchronizes logged RGB-D, camera poses, ANYmal state/contact wrenches, dynamic leg transforms and torque-mode actuator commands into the model's NPZ schema. It uses a preceding frame for causal foot velocity. It requires the synthetic `lcam_front` calibration record and validates alignment before writing. No model architecture or weights are imported.
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

## Observed remote pilot (10 October 2026)

The four source trajectories were found under `/path/to/ugp/V7F_4080_Run/storage/raw_tartanground`. Their raw ROS bags were used only as data; no V7 model source or weights were used. A synthetic camera calibration record was read from `/path/to/ugp/V7_SHORT_CAMERA_CALIBRATION_DATASET.json`. This calibration is specific to the rendered TartanGround `lcam_front`, not a physical ANYmal camera.

The data converter produced four independent 40-frame recordings at 160×160, checked camera/base alignment and contact/TF foot agreement, and passed the model's `audit` command. `data/pilot_causal` contains those recordings on the remote host. The exact supplied model then completed one CUDA epoch with six training windows and two held-out windows, saving `runs/real_data_pilot_causal_20261010/best.pt`. This very small pilot is an execution check; see the status report for logs and limitations. Generated data and checkpoints are ignored by Git and excluded from this source package.

Exact preparation and run commands (executed from this package on the remote host):

```bash
python3 prepare_tartanground_recordings.py \
  --root /path/to/ugp/V7F_4080_Run/storage/raw_tartanground \
  --calibration /path/to/ugp/V7_SHORT_CAMERA_CALIBRATION_DATASET.json \
  --output data/pilot_causal --start 50 --frames 40 --side 160
/path/to/ugp/python terrain_foothold_world_model.py audit --data data/pilot_causal
CUDA_VISIBLE_DEVICES=0 /path/to/ugp/python terrain_foothold_world_model.py train \
  --data data/pilot_causal --out runs/real_data_pilot_causal_20261010 \
  --epochs 1 --batch 1 --limit-per-run 2 --require-cuda
```

The approved two-agent CLI pilot was attempted but its first Codex round failed: the remote network presented an `*.iitk.ac.in` certificate for `chatgpt.com`. No Claude review round ran. TLS verification was not disabled.

## What this setup does NOT claim

- It does not remote-control your PC by itself, authorize SSH, install the CLIs, provide Claude authentication, install CUDA, or start real GPU training.
- It does not turn synthetic execution tests into proof of 3D physical accuracy. Reconstructed surfels are not omniscient true geometry, and point tracking remains uncertain.
- It does not guarantee the starting model fully satisfies the contract. The point of independent review is to surface and fix remaining mismatches before real training.
- It cannot produce contact-force supervision from data lacking a reliable force stream.

## Safety and privacy

Do not paste credentials in a Work conversation, Git repository or prompt. Source code may be sent to each model provider when you run its CLI. Approve access and possible billing before launching. Do not use model output directly as a robot controller without independent safety engineering and validation.
