from pathlib import Path
import csv, json, re

BASE = Path(r'/path/to/ugp/local')
RUN = BASE / 'server_snapshot/path/to/ugp/V7F_4080_Run'
STORE = RUN / 'storage'
parts = []
def add(s): parts.append(s.strip()+'\n')
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def fmt(v, n=4):
    try:
        f=float(v)
        if f != f: return 'N/A'
        return f'{f:.{n}f}'
    except (ValueError,TypeError): return str(v)
def table(headers, records):
    add('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+ '\n'.join('| '+' | '.join(map(str,r))+' |' for r in records))
refs={}
def ref(key,label,path,line=None):
    refs[key]=(label,path,line)
    return f'[{label}][v7-{key}]'

modelref=ref('model','active V7 layers',BASE/'notebook_02_live.txt',2062)
histref=ref('history','causal history encoder',BASE/'notebook_02_live.txt',938)
cfgref=ref('config','active configuration',BASE/'notebook_02_live.txt',516)
specialref=ref('specialist','active specialist layers and fusion',BASE/'notebook_02_live.txt',3185)
lossref=ref('loss','V7 loss implementation',BASE/'notebook_02_live.txt',2353)
flossref=ref('floss','V7-F loss implementation',BASE/'notebook_02_live.txt',3412)
prepref=ref('prep','signal extraction and alignment',BASE/'notebook_02_live.txt',709)
physref=ref('physics','physics identification and derivative preprocessing',BASE/'notebook_02_live.txt',1975)
cacheref=ref('cache','frozen visual cache builders',BASE/'notebook_02_live.txt',1371)
fitref=ref('fit','global and local training',BASE/'notebook_02_live.txt',2412)
ffitref=ref('ffit','force-head training and causal validation',BASE/'notebook_02_live.txt',3480)
gapref=ref('gap','actual gap adaptation implementation',BASE/'notebook_02_live.txt',5356)
unseenref=ref('unseen','zero-shot source-calibration implementation',BASE/'notebook_02_live.txt',6034)
metricsref=ref('metrics','metric definitions',BASE/'notebook_02_live.txt',2577)
ckref=ref('ckpt','saved tensor metadata from original ZIP checkpoints',BASE/'v7_checkpoint_metadata.json')
freshref=ref('fresh-ckpt','fresh server saved tensor metadata',BASE/'remote_model_metadata.json')
origref=ref('old','executed original Colab outputs',BASE/'notebook_04_V7_FULL_4_BRANCH_COLAB_PY313_FIXED (1).txt',4334)
logref=ref('latest','latest RTX execution log',STORE/'training_latest.log',523)
trainlogref=ref('trainlog','RTX training log from 2026-09-28 13:52:35 UTC',STORE/'training_20260928T135235Z.log',1282)
stateref=ref('state','completed RTX runner state',STORE/'training_latest.state.json')
splitref=ref('split','exact 28-trajectory training manifest',STORE/'results/V7_EXPANDED_TRAIN_TRAJECTORIES.csv')
zero_cfg=ref('zero-cfg','strict holdout configuration',STORE/'V7F_unseen_trajectory_zero_shot_2GPU/CONFIG.json')

add(r'''## UGP V7 and V7-F: implemented architecture and experiments

The notebook implements two V7 residual dynamics experts and two additional V7-F force specialists. V7 predicts joint angles, joint velocities, measured torques, contact forces, body signals, contact probabilities, and force-load regimes. V7-F freezes the V7 models and learns corrections to their contact-force vectors. The original Colab experiment, the expanded RTX diagnostic experiment, the temporal-gap experiment, and the strict unseen-trajectory experiment use different data or adaptation protocols; their results are reported separately below.

This section is based on the collected server notebook, source extracts, saved configurations and result CSVs. It is an evidence audit: no training, inference, tests or model-source changes were performed while preparing it. The snapshot contains the runner's final state from 2026-09-28, and the evidence collection is dated 2026-10-05. It does not establish the current live process state.
''')
add(f'Primary sources: {ref("nb","collected server notebook",RUN/"V7F_2xRTX4080_EXPANDED_28TRAIN_4STRICT_HOLDOUT_CHECKPOINTED_FINAL.ipynb")}, {stateref}, {trainlogref}, {logref}.')

add('### Inputs, histories, baselines and targets')
table(['Input','Shape excluding batch','Meaning'],[
    ['P','T × 4 × 12','Per leg: q3, dq3, measured torque3, measured force Fx/Fy/Fz'],
    ['B','T × 12','IMU acceleration3, IMU gyro3, estimated linear velocity3, estimated angular velocity3'],
    ['U','T × 6','Recorded high-level linear/angular twist command'],
    ['L','T × 4 × 3','Recorded low-level commanded joint torque'],
    ['E','T × 4 × 3','At n>0: measured torque[n] − commanded torque[n−1]; first sample uses same-index command'],
    ['Leg history input','4 × T × 24','P12 + U6 + L3 + E3; a shared encoder is applied separately to each leg'],
    ['Body history input','T × 18','B12 + U6'],
    ['Future-control input','H × 36','Last U6/L12 and mean U6/L12 over the recorded anchor-to-horizon control interval'],
    ['Physics input','H × 47','Normalized constant-velocity q12, proxy torque12, proxy acceleration3, force magnitude4, regime one-hot16'],
    ['Foot-query kinematics','4 × 6','Official ANYmal-D URDF foot FK position3 + current linear command × 0.30 s displacement3'],
])
add(f'''SHORT uses 0.5 s context and predicts only +0.1 s. LONG uses 4.0 s context and jointly produces +0.2, +0.3, …, +1.0 s. History length is `round(context_sec / median(diff(time)))`; the original saved positional tensors have 50 and 400 history positions, consistent with 100 Hz samples. Horizon offsets are rounded on the robot time grid. Each inference anchor selects the latest camera frame at or before the anchor and rejects a future visual timestamp. {cfgref}, {ckref}.

The raw baseline is `q_baseline(h) = q_now + h*dq_now`; dq, torque and force remain at their current values, and all body channels remain at the current body value. V7 learns `(target − baseline) / channel_std`, then reconstructs `prediction = baseline + predicted_residual*channel_std`. Binary contact labels use measured force norm **>10 N**; regime labels use **≥10 N** and learned positive-force tertiles. The difference at exactly 10 N is present in the implementation.

Per-run normalization uses channelwise mean/std, with P pooled across time and legs. L uses measured-torque mean/std; E uses measured-torque std without mean subtraction. V7 local adaptation recomputes statistics and soft physics on the latest 20 s through the test anchor. The V7-F specialist preserves the original training-run statistics and physics that its frozen base saw. {prepref}, {fitref}, {ffitref}.

Future recorded U and L are supplied to the horizon heads. Thus the benchmark is conditional on recorded future controller commands. A deployment would need planned commands or another forecasting assumption for these inputs. These are not autonomous forecasts from past sensors alone.''')

add('### Frozen visual feature extraction and trainable visual fusion')
table(['Branch','Foundation encoder actually requested','Cached input to V7','Processing'],[
 ['DINOv3','timm/vit_large_patch16_dinov3.lvd1689m','64 × 1025','1024 visual channels + ground-mask validity; last feature map'],
 ['ConvNeXt','timm/convnext_large.dinov3_lvd1689m','64 × 2881','All four stage features pooled and concatenated (192+384+768+1536=2880), plus mean validity'],
 ['Geometry','Pointcept/Utonia pretrained PTv3','64 × 1234','4096 input points; hierarchical recovered feature1224 + xyz3 + normal3 + RGB3 + roughness1'],
 ['Optical flow','WAFT a1, tar-c-t checkpoint','4 × 64 × 6','Four flows ending at or before the camera frame; channels u,v,magnitude,divergence,curl,confidence'],
])
add(f'''The RGB branches use a geometry-defined ground support crop, resized to 448×448 with ImageNet normalization, and mask-weighted 8×8 pooling. Metric depth is accepted only in the expected encoded/float formats, with usable depth 0.25–8.0 m. Support bounds in the camera-heading, gravity-aligned frame are x=0.15–4.0 m, |y|≤2.5 m, z=−2.0–0.35 m, and normal z≥0.35; the crop has 0.08 margin and requires 500 support pixels. The geometry cloud is sampled to 4096 points; 64 representative points are retained after pretrained encoding. Roughness is `1 − abs(normal_z)`, clipped to the valid range. It is a geometric proxy.

All four foundation models are put in eval mode, frozen, used under inference mode, and unloaded after their float16 caches are written. They are external to the V7 task checkpoints. Their weights do not receive gradients during V7 or V7-F training. The cached flow stack is causal by construction. Its trainable temporal CNN uses two kernel-3, padding-1 convolutions (6→64→128) across the four already observed flows, selects the last output and projects 128→256. Padding does not create access to future images. {cacheref}.

Each branch has a trainable Perceiver resampler: LayerNorm(input), Linear(input→256), GELU, LayerNorm(256), 16 learned latent tokens, and **2** cross-attention/FFN layers. Cross-attention has **8** heads; each FFN is 256→1024→256 with GELU and dropout 0.1. Flow first passes through its temporal CNN, then a resampler with input width 256. The four sets of 16 latent tokens receive modality embeddings and are concatenated into **64×256** visual tokens.

Five terrain queries (one learned global query and four foot queries) pass through **4** fusion blocks. Each block applies 8-head cross-attention to all 64 visual tokens, 8-head self-attention among the five queries, and a 256→1024→256 FFN with dropout 0.1. A foot query is a leg embedding plus an MLP of current normalized proprioception12, command6 and raw FK/displacement6: LayerNorm24→Linear24→256→GELU→Linear256→256. Outputs are a global terrain feature256 and four foot features256. These visual cross-attention and query self-attention operations are unmasked; all tokens are derived from the current or past inputs. {modelref}.

The active V7 classes have **no FiLM layer**. They use query conditioning, cross-attention and concatenation followed by MLPs. They also do not apply explicit FL/FR t−2 versus rear-leg t−3 visual delays; all four foot queries read the same causal visual frame. Such FiLM/delay choices belong to another model implementation in the UGP workspace and must not be attributed to this notebook.''')

add('### Exact V7 task layers and output dimensions')
table(['Module','Layers / dimensions','Result'],[
 ['Shared leg history encoder','Linear24→128; learned positions; 2 Transformer encoder layers; 4 heads; FFN512; GELU; dropout0.1; pre-norm; final LayerNorm','Last history token128 + leg embedding128'],
 ['Body history encoder','Same architecture with input18→128','Last history token128'],
 ['Leg/visual fusion','LayerNorm384; Linear384→256→GELU→Linear256→256→GELU→LayerNorm256','Four fused leg features256'],
 ['Context MLP','LayerNorm1408; Linear1408→768; GELU; dropout0.1; Linear768→384; GELU; LayerNorm384','Context384; input=4×256 + body128 + global visual256'],
 ['Control projection','36→128; GELU; LayerNorm128','128 per horizon'],
 ['Horizon projection','1→64→GELU→128→GELU→LayerNorm128','128 per horizon'],
 ['Physics projection','47→128→GELU→128→GELU→LayerNorm128','128 per horizon'],
 ['Horizon context','Concatenate context384 + control128 + horizon128 + physics128','H × 768'],
 ['Binary contact head','768→256→GELU→4','H × 4 logits'],
 ['Load-regime head','768→256→GELU→16','H × 4 × 4 logits'],
 ['State/body head','768→384→GELU→dropout0.1→36','24 joint q/dq residuals + body12 residuals per horizon'],
 ['Torque/force head','LayerNorm1285; 1285→512→GELU→dropout0.1→256→GELU→6','Per horizon/leg: torque3 + force3 residuals'],
 ['Auxiliary force head','LayerNorm1285; 1285→256→GELU→4','Log magnitude1 + normalized direction3'],
])
add(f'''The 1285-dimensional per-leg head input is horizon context768 + fused leg256 + foot visual256 + contact probability1 + regime probabilities4. Joint q/dq and torque/force residuals are assembled as `(batch,H,4,12)`; body residuals are `(batch,H,12)`. The history encoders use a strictly upper-triangular Boolean attention mask and retain the final token. They share weights across legs but have separate leg identity embeddings. No time mask is used in the horizon-independent V7 output heads. {histref}, {modelref}.

The freshly inspected server state tensors contain exactly **16,314,094** parameters for SHORT and **16,403,694** for LONG, and **2,448,554** for each specialist. The original ZIP checkpoint counts agree. The 89,600-parameter SHORT/LONG difference equals two positional tables ×350 additional positions×128. These counts cover the task models, not the frozen foundation encoders. The module breakdown below was derived from original ZIP tensors; the fresh server tensor totals and dimensions confirm the same task architecture. {freshref}, {ckref}.''')
meta=json.loads((BASE/'v7_checkpoint_metadata.json').read_text())
md=list(meta.values())
table(['Saved task module','SHORT parameter count','LONG parameter count'],[
 [k,format(next(v for key,v in meta.items() if key.endswith('V7_PRIOR_SHORT.pt'))['module_counts'][k],','),format(next(v for key,v in meta.items() if key.endswith('V7_PRIOR_LONG.pt'))['module_counts'][k],',')]
 for k in next(v for key,v in meta.items() if key.endswith('V7_PRIOR_SHORT.pt'))['module_counts']
])

add('### V7-F specialist layers and force reconstruction')
table(['Module','Exact structure'],[
 ['Input projection','LayerNorm1285; 1285→256; GELU; dropout0.10; 256→256; GELU; LayerNorm256'],
 ['Joint-horizon transformer','Two pre-norm Transformer encoder layers; width256; 8 heads; FFN1024; GELU; dropout0.10; applied separately per leg'],
 ['Phase classifier','LayerNorm256; 256→128→GELU→4'],
 ['Four force experts','Each: LayerNorm256; 256→192→GELU→dropout0.08→96→GELU→3'],
 ['Expert gate','LayerNorm265; 265→128→GELU→4; softmax'],
 ['Direct correction','LayerNorm256; 256→192→GELU→3'],
 ['Magnitude/direction correction','LayerNorm256; 256→192→GELU→4'],
 ['Fusion gate','LayerNorm265; 265→96→GELU→1; sigmoid'],
])
add(f'''V7-F extracts detached V7 context `x(batch,H,4,1285)`, raw base force `F_base`, contact probability and regime probabilities. The horizon transformer has **no attention mask**. Its tokens represent the planned conditional predictions at all queried horizons; they contain horizon/control/physics conditioning, not future measured state or force labels. SHORT has one token per leg, LONG has nine.

The phase targets are swing, touchdown, stance and liftoff. They are formed from contact transitions between the input anchor and target horizon, with the label agreeing with the final contact state. The gate input has 256 latent +4 phase probabilities +4 load-regime probabilities +1 contact probability =265 channels. The phase classifier guides a soft expert gate; the implementation does not hard-assign one named phase to one particular expert.

With expert weights `w_e`, per-axis force standard deviation `σ_F`, and MLP corrections `c_e,c_d,m,d`, the two paths and final output are:

`c = Σ_e w_e*c_e + c_d`

`F_direct = F_base + σ_F*c`

`m_base = max(||F_base||, 1e−5)`

`m_new = max(expm1(log1p(m_base) + 0.75*tanh(m)), 0)`

`d_new = normalize(normalize(F_base) + 0.20*tanh(d))`

`F_magdir = m_new*d_new`

`α = sigmoid(fusion_gate); F_final = α*F_direct + (1−α)*F_magdir`.

The correction-output layers and final fusion-gate layer are zero-initialized. Consequently both paths start at the base prediction up to numerical effects and the 1e−5 magnitude floor. Forces stay in the recorded measurement coordinate system: geometry normals are not used to invent an unverified camera-to-base force rotation. {specialref}.

For the diagnostic V7-F CSV, only force rows are replaced in the existing adapted-V7 sensor table. Its joint, torque, body and contact rows are copied from V7. The force specialist itself runs on the **frozen global V7 prior**, not on each full-model locally adapted V7 copy used to construct the earlier diagnostic table. Thus the force comparison measures the complete V7-F pipeline versus the adapted V7 pipeline; it is not simply adding a head to the same locally adapted prediction.''')

add('### Physics proxies and complete training losses')
add(f'''The inverse-dynamics proxy for each leg is the identified regression

`τ_proxy = [ddq, dq, sin(q), cos(q), F, regime_onehot_or_probabilities, 1] W_leg`, with `W_leg` shape20×3.

The body proxy is `acc_proxy = [Σ_legs F, 1] W_body`, with `W_body` shape4×3. Ridge fitting uses λ=0.001 after RMS feature scaling, including its constant column. This is an affine calibrated soft physical consistency proxy, rather than a full rigid-body dynamics model with known mass, inertia and calibrated force frame. Official URDF FK supplies foot-query positions; it is not the inverse-dynamics solver.

The fitting function selects rows in the requested time interval only **after** computing `np.gradient(dq,time)` and per-joint 99.5th-percentile derivative clipping limits over the **entire trajectory**. Central differences at a cutoff can use a later sample, and the clipping scale includes later trajectory data. Actuator-to-joint assignment is also inferred once using the first min(5000,N) aligned samples. These procedures mean that the notebook's word “causal” is stronger than the implemented preprocessing isolation. The gap tests and diagnostics do not establish strictly past-only calibration. {physref}, {prepref}.

Let `Sβ(a,b)` denote Smooth L1/Huber loss with PyTorch beta β, and let normalized residuals be `r̂_P,r_P,r̂_B,r_B`. The q,dq,torque,force and body data losses use beta0.5, average their feature/leg axes, then average horizons with `w_h=linspace(1,1.5,H)` (or1 for SHORT); the weights are not normalized back to mean1. Other terms below use their direct implementation averages. {lossref}.''')
table(['V7 loss term','Coefficient','Exact definition'],[
 ['q','1.00','Horizon-weighted S0.5 of normalized q residuals'],
 ['dq','1.00','Horizon-weighted S0.5 of normalized dq residuals'],
 ['torque','1.50','Horizon-weighted S0.5 of normalized torque residuals'],
 ['force vector','2.00','Horizon-weighted S0.5 of normalized force residuals'],
 ['body','0.50','Horizon-weighted S0.5 of all12 normalized body residuals'],
 ['binary contact','0.25','BCE with logits against measured contact labels'],
 ['force-load regime','0.40','Weighted cross entropy over4 classes'],
 ['auxiliary log magnitude','0.25','S0.25(aux_logmag, log1p(||F_truth||))'],
 ['auxiliary direction','0.10','Contact-weighted mean of1−cos(aux_dir,F_truth_dir); denominator sum(contact)+1e−6'],
 ['kinematic consistency','0.20','S0.5((q_pred−q_now−0.5*h*(dq_now+dq_pred))/σ_q,0)'],
 ['inverse-dynamics consistency','0.20','S0.5((τ_pred−τ_proxy(q_pred,dq_pred,(dq_pred−dq_now)/(h+1e−6),F_pred,regime_probs))/σ_τ,0)'],
 ['Newton/body consistency','0.10','S0.5((acc_pred−acc_proxy(F_pred))/σ_acc,0)'],
 ['zero-force in noncontact','0.08','Mean((1−p_contact)*||F_pred||/(largest regime center+1e−6))'],
 ['long-horizon trend','0.10','S0.5(Δ_hP_pred/σ_P, Δ_hP_truth/σ_P), zero for SHORT'],
])
add('The total V7 objective is the sum of the 14 terms above. Standard-deviation denominators use the code’s 1e−6 safeguards. Regime class weights are inverse square-root class frequency, normalized to mean1 and clipped to[0.5,3.0]. For the expanded RTX run, thresholds are[10,222.11116,273.2027] N, centers[0,195.37396,249.0397,295.5505] N and class counts[618440,221511,221510,221511].')
add(f'''The V7-F total objective is

`L_F = 2.00L_vector + 1.60L_logmag + 0.10L_direction + 0.25L_phase + 0.45L_temporal + 0.35L_integral + 0.015L_anchor + 0.005L_gate`.

Its vector term uses normalized component error and beta0.35, multiplied per horizon/leg by `a = 1 + 1.25*clip(||F_truth||/max(last_regime_threshold,1),0,2) + 0.50*I(phase in touchdown,liftoff)`. The maximum such weight is4.0. {flossref}.''')
table(['V7-F term','Exact definition'],[
 ['Vector','Mean(a * mean_components(S0.35((F_pred−F_truth)/(σ_F+1e−6),0)))'],
 ['Log magnitude','S0.20(log1p(||F_pred||),log1p(||F_truth||))'],
 ['Direction','Mean1−cos on ||F_truth||≥10 N; denominator valid count+1e−6'],
 ['Phase','Unweighted4-class cross entropy'],
 ['Temporal','S0.35(Δ_hF_pred/(σ_F+1e−6),Δ_hF_truth/(σ_F+1e−6)); zero for SHORT'],
 ['Integral','S0.35((Σ_h0.1F_pred−Σ_h0.1F_truth)/(σ_F+1e−6),0); zero for SHORT'],
 ['Anchor','Mean square of direct normalized correction c; does not penalize the entire magnitude/direction path'],
 ['Gate','Mean squared difference between batch-mean expert probabilities and uniform1/4'],
])
add('The “impulse” term is a discrete sum over uniformly spaced horizon predictions. It is not a independently measured impulse target. The variable named gate entropy implements mean-gate squared balancing, rather than entropy. Every coefficient above comes from active code, including the internal0.75/0.20 force reconstruction bounds,1.25/0.50 example weighting and0.1 s integral step.')

add('### Training protocol and what the RTX logs actually executed')
table(['Stage','Configuration','Parameters updated'],[
 ['V7 global SHORT','Seed791; 3 epochs; 0.3 s hop; AdamW lr0.0002; weight decay0.0001; clip1.0; one sample per update','All task-model layers; frozen foundation caches'],
 ['V7 global LONG','Seed801; 4 epochs; 1.0 s hop; same optimizer','All task-model layers'],
 ['V7 diagnostic local adaptation','4 epochs each expert; lr0.00008; latest20 s; half-life5 s SHORT/6 s LONG; chronological windows; test-dependent seed','A fresh full V7 copy per test; local stats/physics refit'],
 ['V7-F global SHORT','Seed7882; 5 epochs; 0.15 s hop; batch16; lr0.0003; AdamW wd0.0001; clip1.0','Force head only; two-GPU DataParallel'],
 ['V7-F global LONG','Seed7883; 6 epochs; 0.25 s hop; batch8; same optimizer','Force head only; two-GPU DataParallel'],
 ['V7-F diagnostic local adaptation','5 epochs; lr0.00006; hops0.3/1.0 s; latest20 s; half-life5/6 s; newest20% windows validation; best validation epoch restored','Force-head copy only; frozen global V7 base; original source-run calibration'],
 ['Same-run gap adaptation','Gaps4,7,9,10 s from latest gradient input anchor; same5 epochs/lr0.00006; no validation split or best-epoch selection','Force-head copy only, independently per gap/test; two GPU workers'],
 ['Strict unseen','No local V7/V7-F updates; no heldout normalization/physics fit; four tests per trajectory','No task parameters updated; cache extraction and frozen inference'],
])
add(f'''The expanded training set has seven trajectories in each of four terrain groups, **28 unique trajectories total**. The four diagnostic trajectories remain part of training. Their global prior cutoff is the first diagnostic time; the other24 training trajectories use cutoff `run_end−1.05 s`. The four unseen trajectories are excluded from both task-model training stages. {splitref}, {zero_cfg}.

Training keys by environment are ForestEnv P2000/P2003/P2007/P2011/P2015/P2019/P2022; GreatMarsh P2000/P2005/P2010/P2015/P2020/P2023/P2029; SeasonalForestWinter P2000/P2003/P2009/P2012/P2015/P2018/P2021; OldTownFall P2000/P2002/P2003/P2004/P2005/P2006/P2007.

The16 diagnostic anchors are ForestEnv/P2000 at28,45,75,90 s; GreatMarsh/P2023 at47,61,73,87 s; SeasonalForestWinter/P2000 at29,46,64,92 s; OldTownFall/P2000 at32,84,100,115 s. Independent V7 local seeds are `781 + terrain_index*1000 + test_id*100 +1` (SHORT) or+2 (LONG), with zero-based terrain index and one-based test id. V7-F starts its seed family at781+7100=7881 and uses+1/+2 for global experts. {cfgref}, {fitref}, {ffitref}.

V7-F global examples are cached detached half-precision contexts and full-precision targets/calibration. Mini-batches draw across trajectories. The sampler balances **force bins**, with duplication of each rare bin capped at2.5× its population; it does not guarantee equal terrain counts within a mini-batch. These densely overlapping windows are optimization samples, not independent trajectories.

The earlier RTX log `training_20260928T135235Z.log` records real global V7 and V7-F optimizer updates and local diagnostic adaptations. V7 prior raw losses per epoch were SHORT1.180507371→0.843577968→0.745650106 and LONG2.446516626→1.702973012→1.472296120→1.322666874. V7-F had1969 batches per SHORT epoch and1540 per LONG epoch, with the executed losses below. The log contains all16 local V7 and16 local V7-F tests. {trainlogref}.''')
table(['Epoch','V7-F SHORT executed loss (1969 batches)','V7-F LONG executed loss (1540 batches)'],[[i+1,a,b] for i,(a,b) in enumerate(zip(['1.5687','1.3267','1.2359','1.1872','1.1313','—'],['2.0814','1.7704','1.6543','1.5783','1.5248','1.4819']))])
add(f'''The **latest** runner starts at2026-09-28 15:56:44 UTC and finishes at16:45:46 UTC with status`complete` and code cells1–23 completed. It logs two NVIDIA GeForce RTX4080 GPUs. It does **not** rerun the global training epochs or the16 original local diagnostics: these come from persisted checkpoints/results. The log prints an RNG restoration error (“RNG state must be a torch.ByteTensor”) after loading each model, optimizer and completed epoch. In the active code, these assignments precede RNG restoration inside a broad try block, and the exception does not undo them. The subsequent epoch loops therefore execute zero epochs; the completed diagnostic signatures match and all16 V7 and16 V7-F tests are skipped. This code trace and the absence of global epoch updates explain the very short V7/V7-F cell times; the warning is not evidence of training from scratch. It also means exact RNG restoration was unsuccessful in this rerun. {logref}, {fitref}, {ffitref}, {stateref}.

The latest run actually executes all64 same-run gap jobs (16 tests×4 gaps), with a SHORT and LONG5-epoch head adaptation in each job; it then extracts features for the four strict holdouts and performs16 zero-shot tests. Gap cell22 takes130.8 s; unseen cell23 takes2703.9 s including download/preprocessing. The unseen timing CSV reports0.218205 s for all zero-shot inference with caches prepared, and heldout preprocessing about657–687 s per trajectory. This cached inference timing does not include the foundation encoders or establish real-time end-to-end deployment speed. {ref("timing","saved unseen timing",STORE/"V7F_unseen_trajectory_zero_shot_2GPU/TIMING.csv")}.

Optional V5 checkpoint paths are absent in the RTX log. V5 numerical predictions are not training targets in these V7/V7-F stages. The eight stored “best existing V5” values below are hard-coded diagnostic evaluation references.''')

add('### Metric definitions and aggregation')
add(f'''Force **component MAE** is the mean absolute error across Fx/Fy/Fz, then averaged over vectors. Force **vector error** is `||F_pred−F_truth||₂` in N; **magnitude error** is `abs(||F_pred||₂−||F_truth||₂)` in N. Direction cosine is reported only where ground-truth force meets the contact cutoff (and, in the gap/unseen functions, predicted magnitude exceeds1e−9); noncontact directions are N/A. “Vector error percent of mean force” is100×mean vector error/mean true force magnitude, including all vectors. It is a ratio of means and differs from averaging per-vector or per-test percentages.

Each diagnostic/unseen horizon has16 test anchors×4 legs=64 force vectors; each unseen terrain/leg mean has4 anchors. The ten horizons are conditional predictions at the same16 anchors, rather than160 independent trajectories. Diagnostic sensor-group RMSE is computed inside each test/horizon/group and then averaged across16 tests. NRMSE divides that within-test RMSE by the group’s within-test ground-truth std plus1e−6, then averages. It is not a global pooled RMSE or a force-relative percentage.

Contact prediction is thresholded at0.5; accuracy/precision/recall/F1 use the64 leg labels per horizon, Brier is mean squared probability error and AUROC uses probability ranking when both classes occur. Trend correlation is Pearson correlation over the nine LONG horizons for each sensor/test, averaged by group; first-difference sign accuracy is the fraction of adjacent horizon changes whose signs agree. {metricsref}.''')

add('### Original executed Colab diagnostic results (historical, distinct from RTX)')
add(f'The executed original notebook retains the following force comparison over its familiar four trajectories with local adaptation. These numbers are historical Colab outputs and are **not** the expanded RTX CSVs. {origref}.')
table(['Horizon s','V7 component MAE N','V7-F component MAE N','Reduction %'],[
 [a,b,c,d] for a,b,c,d in [
 ['0.1','6.941602','6.077722','12.444966'],['0.2','18.441826','10.539195','42.851674'],['0.3','20.194385','13.798430','31.671947'],['0.4','15.052799','9.208241','38.827047'],['0.5','18.171151','11.492774','36.752636'],['0.6','17.251262','11.916269','30.925237'],['0.7','19.692355','13.657293','30.646725'],['0.8','16.426138','9.799497','40.342055'],['0.9','22.386615','14.078678','37.111181'],['1.0','17.682948','10.967694','37.975875']]])
add('Original V7 vector errors were14.502486 N at+0.1 s and38.374624 N at+1.0 s. Comparing those numbers directly to the expanded unseen experiment changes both the training set and adaptation protocol, so it does not constitute a controlled improvement estimate.')

add('### Expanded RTX diagnostics: V7 versus V7-F')
comp=rows(STORE/'V7F_force_specialist/V7F_VS_V7_FORCE_BY_HORIZON.csv')
v7vec=rows(STORE/'results/V7_FORCE_VECTOR_METRICS_BY_HORIZON.csv')
fvec=rows(STORE/'V7F_force_specialist/V7F_FORCE_VECTOR_METRICS_BY_HORIZON.csv')
table(['Horizon s','V7 component MAE N','V7-F component MAE N','Reduction %','V7 vector error N','V7-F vector error N','V7-F magnitude error N','V7-F cosine'],[
 [fmt(c['horizon_s'],1),fmt(c['V7_force_MAE']),fmt(c['V7F_force_MAE']),fmt(c['improvement_percent'],2),fmt(v['force_vector_error']),fmt(f['force_vector_error']),fmt(f['force_magnitude_MAE']),fmt(f['force_direction_cosine'],6)] for c,v,f in zip(comp,v7vec,fvec)])
add(f'Sources: {ref("forcecomp","RTX force comparison",STORE/"V7F_force_specialist/V7F_VS_V7_FORCE_BY_HORIZON.csv")}, {ref("v7vec","RTX V7 vector metrics",STORE/"results/V7_FORCE_VECTOR_METRICS_BY_HORIZON.csv")}, {ref("fvec","RTX V7-F vector metrics",STORE/"V7F_force_specialist/V7F_FORCE_VECTOR_METRICS_BY_HORIZON.csv")}.')

sensor=rows(STORE/'results/V7_MEAN_METRICS_ACROSS_16_TESTS.csv')
add('The following V7 sensor MAEs are also present in the V7-F diagnostic sensor table because the specialist replaces only force rows. Units: q rad, dq rad/s, torque N·m, acceleration m/s², gyro and body angular velocity rad/s, body linear velocity m/s. “Contact” is probability absolute error.')
groups=['q','dq','tau','acc','gyro','body_linear_velocity','body_angular_velocity','contact']
table(['Horizon s','q','dq','Torque','Acc','Gyro','Body linear v','Body angular v','Contact'],[
 [fmt(h,1)]+[fmt(next(r for r in sensor if float(r['horizon_s'])==h and r['signal_group']==g)['MAE']) for g in groups] for h in [round(i/10,1) for i in range(1,11)]])
add(f'{ref("sens","full RTX sensor MAE/RMSE/NRMSE and baseline metrics",STORE/"results/V7_MEAN_METRICS_ACROSS_16_TESTS.csv")} retains every group/horizon and baseline value. Endpoint comparisons to the recorded V5 reference are:')
acc=rows(STORE/'V7F_force_specialist/V7F_ACCEPTANCE_VS_BEST_EXISTING.csv')
v7acc=rows(STORE/'results/V7_ACCEPTANCE_VS_BEST_EXISTING.csv')
table(['Horizon s','Group','Recorded V5 reference','RTX V7 MAE','RTX V7-F MAE'],[[fmt(a['horizon_s'],1),a['group'],fmt(a['best_existing_V5_MAE']),fmt(v['V7_MAE']),fmt(a['V7_MAE'])] for a,v in zip(acc,v7acc)])
add(f'All eight stored reference comparisons pass the code’s strict lower-MAE check. This evaluates fixed familiar diagnostics and does not establish terrain generalization. {ref("accept","RTX V7-F acceptance reference table",STORE/"V7F_force_specialist/V7F_ACCEPTANCE_VS_BEST_EXISTING.csv")}.')
contact=rows(STORE/'results/V7_CONTACT_METRICS_BY_HORIZON.csv')
table(['Horizon s','Accuracy','Precision','Recall','F1','Brier','AUROC'],[[fmt(r['horizon_s'],1)]+[fmt(r[k],6) for k in ['accuracy','precision','recall','F1','Brier','AUROC']] for r in contact])
add(f'{ref("contact","RTX contact metrics",STORE/"results/V7_CONTACT_METRICS_BY_HORIZON.csv")}. The pooled4-regime confusion matrix below contains640 predictions (16 tests×10 horizons×4 legs):')
cm=rows(STORE/'results/V7_4REGIME_CONFUSION_MATRIX.csv')
table(['True regime','Pred no','Pred light','Pred medium','Pred heavy'],[[r['']]+[r[k] for k in ['pred_0','pred_1','pred_2','pred_3']] for r in cm])
add(f'Correct predictions586/640=91.5625%. {ref("regime","RTX regime confusion matrix",STORE/"results/V7_4REGIME_CONFUSION_MATRIX.csv")}.')
trend=rows(STORE/'results/V7_LONG_HORIZON_TREND_METRICS_MEAN.csv')
table(['LONG signal group','Mean Pearson correlation','First-difference sign accuracy'],[[r['signal_group'],fmt(r['correlation'],6),fmt(r['first_difference_sign_accuracy'],6)] for r in trend])
add(f'{ref("trend","RTX long-horizon trend metrics",STORE/"results/V7_LONG_HORIZON_TREND_METRICS_MEAN.csv")}. High contact classification accuracy and lower contact trend-sign accuracy measure different properties; exact sign agreement is sensitive to nearly constant probabilities and ties.')

add('### Same-run temporal-gap force experiment')
add(f'''For each original test anchor and requested gap g, the latest **local gradient input anchor** is t_test−g. The latest target label can be g−0.1 s before the test for SHORT and g−1.0 s before it for LONG. For example, gap4 s at test28 s uses input24.0 s, with latest labels24.1 s (SHORT) or25.0 s (LONG). All local labels remain before the test, and the audit CSV records actual anchor/target gaps. The specialist trains on all permitted local windows for five epochs; the final head is evaluated. {gapref}, {ref("gap-audit","actual anchor and target-gap audit",STORE/"V7F_same_run_gap_sweep_2GPU/SAME_RUN_GAP_AUDIT.csv")}.

The saved configuration says no training or validation/model-selection data follows the cutoff, but the source shows a narrower restriction: the **local input anchors** obey the cutoff; their target labels can follow it, and the frozen global priors/source normalization/physics are not retrained to that cutoff. At the first diagnostic test, prior calibration reaches the test anchor itself. Thus this experiment is a gap in additional local adaptation, not a fully isolated gap in all training data or preprocessing.''')
gap=rows(STORE/'V7F_same_run_gap_sweep_2GPU/SAME_RUN_GAP_SWEEP_OVERALL.csv')
table(['Requested input gap s','Horizon s','Component MAE N','Vector error N','Magnitude error N','Direction cosine','Vector error / mean true force %'],[[fmt(r['requested_anchor_gap_s'],0),fmt(r['horizon_s'],1),fmt(r['mean_component_MAE_N']),fmt(r['mean_vector_error_N']),fmt(r['mean_force_magnitude_abs_error_N']),fmt(r['mean_direction_cosine'],6),fmt(r['vector_error_percent_of_mean_force'],2)] for r in gap])
add(f'Each row averages64 vectors. {ref("gap-result","RTX gap overall results",STORE/"V7F_same_run_gap_sweep_2GPU/SAME_RUN_GAP_SWEEP_OVERALL.csv")}; {ref("gap-leg","complete gap results by terrain and leg",STORE/"V7F_same_run_gap_sweep_2GPU/SAME_RUN_GAP_SWEEP_BY_TERRAIN_LEG.csv")}. Errors do not increase monotonically with input gap. This small, familiar-trajectory diagnostic does not identify a universally preferred gap.')

add('### Strict unseen-trajectory zero-shot force experiment')
add(f'''The excluded holdouts are ForestEnv/P2001, GreatMarsh/P2027, SeasonalForestWinter/P2006 and OldTownFall/P2001. Each is tested at20,27,36,46 s, producing spacing7,9,10 s. Training/holdout key overlap is empty. Both V7 experts and V7-F global heads are frozen: there are no heldout gradient updates, no local adaptations, and no fitting of normalization or physics on heldout runs. Each terrain borrows the corresponding original training-run calibration. {zero_cfg}, {unseenref}.

The run holds out trajectories within the same four environments used for training. It is evidence of unseen-trajectory transfer, not unseen-terrain-category transfer. Heldout sensor histories and current measured force remain valid observed inputs, while future recorded controls remain supplied as described above. The upstream actuator assignment and source physics preprocessing limitations also remain relevant.''')
zero=rows(STORE/'V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_OVERALL.csv')
table(['Horizon s','Vectors','Component MAE N','Vector error N','Mean true force N','Magnitude error N','Direction cosine','Vector error / mean true force %'],[[fmt(r['horizon_s'],1),r['n_vectors'],fmt(r['mean_component_MAE_N']),fmt(r['mean_vector_error_N']),fmt(r['mean_true_force_mag_N']),fmt(r['mean_force_magnitude_abs_error_N']),fmt(r['mean_direction_cosine'],6),fmt(r['vector_error_percent_of_mean_force'],2)] for r in zero])
add(f'{ref("zero-result","RTX strict unseen overall results",STORE/"V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_OVERALL.csv")}. At+0.1 s the force-vector error is18.7853 N; at+1.0 s it is13.4419 N. The nonmonotonic curve should be reported as measured rather than smoothed into a worsening-with-horizon claim.')
leg=rows(STORE/'V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_BY_TERRAIN_LEG.csv')
add('Endpoint detail by heldout terrain and leg follows. Every cell averages four tests; direction cosine excludes undefined noncontact directions.')
table(['Terrain','Horizon s','Leg','Component MAE N','Vector error N','Mean true force N','Magnitude error N','Cosine','Relative vector error %'],[[r['terrain_type'],fmt(r['horizon_s'],1),r['leg'],fmt(r['mean_component_MAE_N']),fmt(r['mean_vector_error_N']),fmt(r['mean_true_force_mag_N']),fmt(r['mean_force_magnitude_abs_error_N']),fmt(r['mean_direction_cosine'],6),fmt(r['vector_error_percent_of_mean_force'],2)] for r in leg if float(r['horizon_s']) in (0.1,1.0)])
add(f'{ref("zero-leg","complete unseen terrain/leg results at all horizons",STORE/"V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_BY_TERRAIN_LEG.csv")}, {ref("zero-test","complete unseen results by test",STORE/"V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TRAJECTORY_BY_TEST.csv")}, {ref("zero-manifest","unseen test manifest",STORE/"V7F_unseen_trajectory_zero_shot_2GPU/UNSEEN_TEST_MANIFEST.csv")}. A given leg can have excellent direction and sizable magnitude error; direction alone is insufficient to characterize force quality. Four observations per terrain/leg also limit how strongly to interpret an individual percentage.')

add('### Foundation-model dimensions and serialized weight totals')
ext=STORE/'.work/v7_external'
add(f'''Fresh safetensors headers and CPU checkpoint metadata establish the following **serialized tensor-element totals**. Foundation totals can include buffers and unused heads. They are not counts of active UGP trainable parameters, and WAFT's checkpoint already contains its Depth-Anything component, so adding the separate Depth-Anything checkpoint would double-count that component. {freshref}.''')
table(['Serialized foundation checkpoint','Tensor count','Tensor-element total','Verified shape / depth evidence'],[
 ['DINOv3 ViT-L/16','318','303,079,424','24 Transformer blocks; width1024; QKV3072×1024; FFN4096; task uses last feature map'],
 ['DINOv3 ConvNeXt-L','342','196,230,336','Stage depths3/3/27/3; widths192/384/768/1536; depthwise7×7 kernels; MLP expansion4×'],
 ['WAFT a1 tar-c-t','604','64,154,559','Entire serialized flow model, including pretrained feature components'],
 ['Depth-Anything-V2 ViT-S checkpoint','239','24,785,089','Separate initializer/component checkpoint; not an additional V7 visual branch'],
 ['Utonia pretrained PTv3','477','137,253,816','Encoder depths3/3/3/12/3; widths54/108/216/432/576; embedding input9; no decoder tensors'],
])
add(f'''The DINOv3 HF config identifies `vit_large_patch16_dinov3`, `num_features=1024`, `num_classes=0`, and flexible input size. The task explicitly supplies448×448 crops. Its checkpoint shape index verifies24 blocks and FFN4096. The stored config/header does not encode the attention-head split, so the exact head count is not independently verified here. ConvNeXt stage depths and widths above come directly from its tensor keys and shapes.

The collected WAFT `tar-c-t.json` selects `dav2_backbone=vits`, `network_backbone=vits`, `algorithm=waft-a1`, five iterative updates, uncertainty range0–10 and `use_var=true`. The collected constructor shows the Depth-Anything-V2 Small feature model with features64 and output widths[48,96,192,384], a ResNet18 deconvolution feature pyramid with widths64/128/256/512, a ViT refinement branch with patch8 and intermediate indices[2,5,8,11], and six flow-head outputs: flow2 plus mixture weights2 and log-scale2. These are cache-extractor details; its dataset/batch/lr/200000-step fields describe the supplied pretrained WAFT training configuration, not UGP task training. {ref("waft-config","collected WAFT pretrained config",ext/"WAFT/config/a1/tar-c-t.json")}, {ref("waft-model","collected WAFT a1 constructor",ext/"WAFT/model/waft_a1.py")}, {ref("waft-vit","collected WAFT refinement transformer",ext/"WAFT/model/backbone/vit.py")}.

Utonia's worker explicitly overrides encoder patch sizes to[1024,1024,1024,1024,1024] and disables Flash Attention. It uses the official default transform with scale0.5, no enforced positive z and no coordinate normalization. Fresh state shapes verify encoder widths[54,108,216,432,576] and depths[3,3,3,12,3]. Each stage's saved RoPE inverse-frequency vector has length3; the collected RoPE formula and compatible channel divisibility imply head dimension18, hence heads[3,6,12,24,32]. This is a shape/source derivation rather than a quoted checkpoint-config field. MLP matrices expand widths4×. There are no saved decoder tensors. The geometry worker concatenates recovered stage features216+432+576=1224, then appends10 geometric/color channels, yielding the actual1234-wide cache. The source constructor's different default widths[48,96,192,384,512] must not replace these loaded-weight dimensions. {freshref}, {ref("utonia-model","collected Utonia class and checkpoint loader",ext/"Utonia/utonia/model.py",646)}.

These frozen weights are loaded sequentially for cache extraction and are not resident trainable branches inside the saved task models. A sum of serialized checkpoint elements would neither establish simultaneous model memory use nor identify the exact number of active parameters in an end-to-end deployment.''')

add('### Interpretation supported by the artifacts')
add('The expanded RTX diagnostic CSVs show lower force component MAE for V7-F at every horizon compared with the adapted V7 pipeline, while the copied V7 non-force rows retain their reported errors. The strict unseen CSVs establish a separate zero-shot result on four excluded trajectories, with18.7853 N vector error at+0.1 s and13.4419 N at+1.0 s. These results do not establish generalization to unseen terrain categories, a fully causal preprocessing protocol, real-time deployment performance, or a matched causal ablation of every training/adaptation choice. The final runner’s completed state confirms its23 cells finished; earlier logs provide the actual global training evidence and the latest log provides the gap/unseen execution evidence.')

for key,(label,path,line) in refs.items():
    url=path.as_posix()
    if line: url += ':'+str(line)
    parts.append(f'[v7-{key}]: <{url}>\n')
out=BASE/'v7_architecture_evidence.md'
out.write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps({'file':str(out),'bytes':out.stat().st_size,'lines':len(out.read_text(encoding='utf-8').splitlines()),'source_refs':len(refs)}))
