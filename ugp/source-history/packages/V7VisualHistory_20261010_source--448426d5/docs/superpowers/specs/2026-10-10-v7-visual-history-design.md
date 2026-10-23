# V7: visual history aligned with robot history

Status: implementation authorized by the user on 2026-10-10. Training is not requested in this turn.

## Purpose and verified starting point

Include past DINO RGB, ConvNeXt RGB and geometry encodings alongside the current encodings. Relate changes in observations to changes in measured robot state, and allow each leg to retrieve terrain seen before its contact with that terrain.

The saved server robot timestamp arrays checked on 2026-10-10 have a 0.01 s interval in all 32 cached trajectories (28 training and four excluded). The existing short branch uses 50 samples, including the current sample: t-0.49 to t, predicting +0.1 s. The long branch uses 400 samples: t-3.99 to t, predicting +0.2 through +1.0 s. These are nominal 0.5 s and 4 s windows. The dataset documents 10 Hz camera collection; individual cached camera intervals still need verification before implementation.

At each synchronized robot timestamp, P contains four legs times 12 measurements: three joint angles, three joint speeds, three measured torques and three foot-force components per leg. B contains 12 body measurements: acceleration3, gyro3, linear velocity3 and angular velocity3. This is 60 measured scalar entries per timestamp, not 60 necessarily new independent raw sensor readings: slower streams are causally held onto the IMU time grid.

Additional unique input fields are high-level commands6, commanded joint torques12 and derived torque tracking errors12, giving 90 distinct scalar fields. The leg encoder receives 24 values per leg/time and the body encoder receives 18 values per time; the shared command6 is repeated between these modules. Gyro and angular velocity have separate fields but both describe rotational motion, so the body12 count does not imply 12 independent physical degrees of freedom.

## Image sequence and timing

Use the current available camera frame and its five immediately preceding available frames, six frames total. At the documented normal cadence this spans approximately 0.5 s. Preserve actual frame times and IDs. Do not describe these six consecutive frames as four seconds of visual coverage. Missing frames may increase the span; duplicated inputs must not masquerade as extra observations.

Use the same recent visual-history policy initially for short and long, preserving their existing full robot-history windows. This leaves a possible limitation: terrain encountered after more than 0.5 s may have fallen outside the visual memory. A longer visual-history experiment should follow measurement of the relevant terrain-to-contact delays; its frame policy must be stated separately.

## Proposed data flow

1. Reuse cached DINO, ConvNeXt and geometry features for the six selected images; use the existing resamplers with shared weights across time. Keep the current optical-flow history.
2. Retain time-indexed outputs of the existing four leg-history encodings and body-history encoding instead of exposing only their final summaries to temporal fusion. Preserve the existing final summaries for the original V7 route.
3. Keep every time-indexed encoded robot sample in memory: four leg tokens and one body token per sample. The user's follow-up approved attention learning the time relationship, so do not impose image-to-robot interval assignments or reduce the robot sequence to raw averages.
4. Add the actual relative image timestamp to each visual token and actual relative robot timestamp to each robot token. Missing images are masked explicitly. Both streams use seconds relative to the same prediction anchor.
5. Let the current global/body and four leg queries read the timestamped memory using the existing query-fusion blocks. Retain spatial visual tokens so different legs can select different regions and earlier times. This is learned retrieval, not a hard rule that an image can affect only the simultaneous robot sample.
6. Keep the existing V7 prediction heads, physical-unit reconstruction, horizon inputs and supervised losses for the first comparison. Evaluate changes to downstream V7-F or joint specialists only after regenerating their input features from this revised foundation.

Six frames times three resampled branches times 16 tokens gives 288 visual tokens. Adding the current 16 flow tokens gives 304. Five robot tokens per time add 250 short or 2000 long: memory size 554 short or 2304 long, width256. Only five current queries read this memory; there is no self-attention across all 2304 memory tokens. Reuse the original four query-fusion blocks. A zero-initialized output correction preserves original V7 predictions at initialization. Freeze the pretrained foundation by default and train only the small history adapters. This is a design budget, not a measured runtime. Reusing pretrained image encodings avoids running the large image backbones again for every overlapping training window.

## Terrain delay and spatial identity

Earlier vision describes terrain that may influence the robot when a foot reaches it; vision itself does not generate the measured force or torque. Time offsets alone do not prove two image patches describe the same ground location. Delay depends on gait, speed, leg and terrain position. Body motion also changes the image, so a learned visual association is not a causal guarantee.

Preserve patch/geometry coordinates. If observed camera/body pose estimates are available, a separately declared geometric alignment can transform past terrain into a common frame. Do not assume metric points from separate camera frames are already in the same frame. Any new pose input and its source must be declared; future poses or states are forbidden.

## Evaluation and implementation boundary

Compare latest-image V7 against this visual-history V7 on identical trajectory identities, eligible optimization ranges, known-control assumptions, losses and per-horizon tests. Report short and long separately, including HAA/HFE/KFE errors and each leg's force/foot-position errors. Rebuild context-dependent caches with new signatures; old raw per-image feature caches may be reused, but old frozen V7 context caches cannot represent added history. No improvement or 2% accuracy is assumed.

Required checks before training are timestamp causality, correct six-frame selection, irregular/missing-frame handling, shared clock, absence of future target inputs, exact warm-start equivalence and checkpoint/cache provenance. They are checks of consequential temporal behavior, not generic smoke tests. Preserve the existing V7/V7-F and joint-specialist files and checkpoints; implement in a new package and separate run directory.
