# Pretrained leg/body-to-force feedback

This is a new isolated experiment. The completed parent and synced sources are unchanged.

## Architecture

The fixed V7/V7-F foundation runs once. A fixed parent V7Legs/V7B hybrid supplies the initial legs/body. A trainable copy reuses its joint-best weights. Zero adapters introduce previous leg/body estimates. A new force transformer consumes learned leg/body tokens, corrected physical state, geometry, previous force, original V7 context, contact and horizon. All three refinement rounds share weights.

Leg/body candidates reference original V7; force candidates reference original V7-F. Learned blending gates start at .25, stay <= .5, and force corrections are bounded to two training standard deviations per axis. Zero additions preserve the existing prediction before training. The loop revises the same target time; it does not move further into the future or guarantee monotonically improving accuracy.

Stage A trains only one force addition. Stage B tunes the new force model, adapters and reused leg/body weights with supervised losses at all three rounds. Frozen seed and original foundation receive no gradients. Actual future values are targets only. Body remains 12 acc/gyro/v/omega channels.

## Data and results

Exactly the parent 28 training trajectories/four disjoint holdouts and short/long splits are enforced. Parent latent caches, masks and normalizations are reused; actual force is attached at exact stored target times and verified against saved gtF. New force scales use optimization windows only.

Resumable best/latest checkpoints contain model, optimizer, RNG, sampling order, cursor, patience and hash provenance. Validation selects checkpoint and round. After training, automatic evaluation exports seed, independent one-pass and recurrent rounds 1/2/3 across 96 cases (16 familiar, 64 gap, 16 unseen), +.1 through +1 s, all legs and joints. The original V7/V7-F local adaptation is shared across variants; correction heads remain fixed for all test cases. Thus familiar/gap values are a matched new experiment, not identical to prior exports that locally trained leg/body heads. No unseen optimization or fitting occurs.

Each workbook has Joint exact, Body exact, Force exact and How to read sheets. Force export uses the updated force, not the original frozen value. Percentages divide absolute error by absolute actual; zero is undefined. Cosines are undefined for zero vectors and displayed with six decimals.

## Deployment

Expected layout:

```
V7Feedback_20261007/
  code/v7feedback/       # this directory, excluding .verification_deps
  code/v7legs_body/      # unchanged parent code copied locally on server
  results/
```

From an existing connected SSH terminal, after checking GPU occupants:

```bash
RUN=/path/to/ugp/V7Feedback_20261007
GPU=1 nohup bash "$RUN/code/v7feedback/run_training.sh" > "$RUN/training.log" 2>&1 < /dev/null &
echo $!
```

Live stats:

```bash
watch -n 2 'cat /path/to/ugp/V7Feedback_20261007/results/status.json; nvidia-smi'
```

Material CPU checks: run test_feedback.py with the existing parent code on the import path. A local isolated CPU PyTorch can be used; it is not included in the delivery. GPU trainer additionally validates real force alignment, seed equality and a real force-loss backward pass before training.

The GPU launch is not established until a matching live PID and optimizer-step log are observed. SSH port failure is not evidence about remote training state.
