# Paste this entire request into ChatGPT Desktop → Work (or Codex) on your Windows machine

**Important: This is the NEW 3D Terrain–Foothold World Model project, NOT V7. Do not import old V7 code, checkpoints, training history, or chat assumptions.**

Use the Computer Use integration and the necessary local/Remote-SSH access with my approval. I have supplied the `Terrain_Foothold_Collaboration.zip` artifact. My project involves a Windows PC, VS Code Remote-SSH and a mentor-provided RTX 4080 Linux server. I want you to **set up and validate a bounded, independent two-agent coding collaboration** for this project:

1. Extract the ZIP into a NEW project folder on the remote server, not in any old V7 repository. If the ZIP is currently only on Windows, use VS Code Remote-SSH or SCP to transfer it. Verify that terminal commands run on the remote host with `hostname`, `pwd`, `nvidia-smi`, and Python's `torch.cuda.is_available()`; do not assume having a VS Code SSH window automatically makes all desktop commands remote.
2. Read `ARCHITECTURE_CONTRACT.md`, `README.md`, `terrain_foothold_world_model.py`, and `two_agent_collab.py` in full. Inspect the script before running it. The model is a research starter, not a physically validated architecture; no previous weights may be reused.
3. Check `codex --version`, `claude --version`, `python --version`, `git --version`, and CLI authentication status on the REMOTE host. Configure Codex CLI as the implementing agent, Claude Code CLI as a separate **read-only reviewer**. If Claude access is unavailable, ask whether to use a second isolated Codex session as the reviewer. DO NOT ask me to paste API keys into chat or terminal output; follow official local sign-in flows.
4. Ask for my approval **before installing packages, signing in, transmitting my source code to any model provider, committing files, or launching agents**. Tell me which external model services will receive code and about any additional billing. Do not enable bypass-permissions modes.
5. Initialize a dedicated Git repository and baseline commit (after approval); ensure the working tree is clean. Run `python two_agent_collab.py --check` and resolve real setup failures. Do not alter this script to fake passing checks.
6. After authorization, launch a short two-agent pilot:
   `python two_agent_collab.py --run --reviewer claude --rounds 2 --smoke "python terrain_foothold_world_model.py self-test"`
   Monitor the local reports under `.collab_runs/`. Stop on failures or human-action blockers; do not silently continue.
7. Have Codex IMPLEMENT missing details of the explicit evolving-3D-terrain → future foothold → force/proprio computation graph. Have the independent reviewer find blind shortcuts, mistaken temporal correspondence, force-label fabrication, leakage, egomotion/deformation confusion, safety issues, and real-data blockers. Pass review back to Codex. Show the Git diff, tests and unresolved issues. **Do not claim model accuracy, completeness or successful GPU training without real evidence.**
8. Ask separately before accessing confidential robot bags or beginning real GPU training. Verify camera calibration, physical contact/force sensors, and data provenance. If real labels are unavailable, explain the limitation; never invent them.

**Outcome I want**: a working two-agent collaboration launched from my RTX 4080 remote VS Code environment, independently reviewing/implementing THIS new model; exact status of the CLI setup, remote host, test results, changed files, remaining blockers, and next required approval. Do not call it V7.
