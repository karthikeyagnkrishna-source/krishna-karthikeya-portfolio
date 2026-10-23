from pathlib import Path
import json
import csv
import io

root=Path(__file__).resolve().parent
out=root.parents[1]/'outputs'
out.mkdir(exist_ok=True)
d=json.loads((root/'remote_evidence_bundle.json').read_text(encoding='utf-8-sig'))
meta=json.loads((root/'feelwm_checkpoint_metadata.json').read_text())
BASE='/path/to/ugp/merged_feelwm_v7_runs/'
def j(path):return json.loads(d['files'][path]['text'])
def f(v):
    if v is None:return 'Unavailable'
    return f'{v:.8g}' if isinstance(v,(int,float)) else str(v)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(f(v) for v in row)+' |' for row in rows])+'\n'
s=['# FeelWM / TartanGround experiment records and results\n',
'These records follow the architecture section above. `K=1` predicts one 0.1 s transition; `K=5` predicts five successive transitions through 0.5 s. A pilot cap is an optimizer-step budget. The ten-epoch study is a separate run using the same prepared data.\n']
ledger=[];detail=[]

s += ['## Engineering probes\n',
'`full_probe` completed one optimizer step with the full 110,526,384-parameter baseline under **DDP, world size two, BF16**, according to its log. Its recorded training total was **24.4411125183** and elapsed launcher timing **4.97 seconds**. Inputs were synthetic integration data, so this is proof of execution/backpropagation plumbing, not a robotics generalization result.\n',
'`tiny_integration` retained one training step (total **24.5023097992**) and one validation event (total **25.9814872742**). Its saved configuration uses a reduced model. A corresponding GPU-identifying log was not retained in this collected record; it is listed as an engineering artifact rather than asserted as a separate scientific RTX result. Neither probe supplies physical test metrics.\n']

arc=d['archives'][BASE+'tartanground_pilot/exports/tartanground_p2000_k1_results_step200.zip']
p2000=json.loads(arc['selected_files']['run/validation_summary.json']['text'])
s += ['## Single-trajectory P2000 K1 pilot\n',
'**Architecture:** the earlier baseline merged model, 110.53 million trainable parameters, with the earlier loss implementation preserved in the P2000 archive. The later per-leg/K5 implementation is audited separately.\n',
'**Data/training:** ForestEnv/P2000, 1,200 camera frames over 119.9 s; 834 training samples and 238 later validation samples, separated by a 5 s purge. Two RTX 4080 GPUs with DDP; per-GPU batch one, accumulation four, approximately eight examples per optimizer update; BF16 and TF32. The retained result summary records **200 optimizer steps**, approximately **1.91 passes** through the training examples and approximately **592 s** for the successful pipeline. The metrics stream has an earlier repeated step-one attempt; the final checkpoint/result is the step-200 lineage.\n',
'**Evaluation:** validation events at steps 50/100/150/200, capped to 50 batches per GPU (100 samples per event). This is later-time validation within P2000. It is not an unseen-trajectory test. No physical-unit RMSE, accuracy, precision, recall or F1 was exported.\n',
table(['Step','Total loss','DINO MSE','Conv MSE','Geometry MSE','Flow MSE'],[[x[k] for k in ['step','total','dino','conv','geom','flow']] for x in p2000['validation']]),
'Final recorded validation components:\n',table(['Loss component','Value'],[[k,v] for k,v in p2000['validation'][-1].items() if k not in {'type','step'}]),
'The two image-latent losses dominate this composite objective. A value such as `v7f_vec=0.96297888` is a standardized training term; it is not 0.96 N force error. The checkpoint SHA-256 recorded by its export is `c463db53de0c1fedef9803bd150dc6d56b2d2c298856e37314b964afc71f671e`. The frozen visual-resampler checkpoint SHA-256 is `a7170559a55cf4cfa43c25ea2778be1da834df66fca17d61462c220d67fdcd4e`. Those are recorded export provenance, not newly recomputed multi-gigabyte hashes in this audit.\n']
ledger.append({'experiment':'P2000 baseline K1 pilot','status':'completed capped pilot','parameters':110526384,'optimizer_steps':200,'epoch_note':'approximately1.91','test_scope':'later-time P2000 validation only','best_val_total':p2000['validation'][-1]['total'],'source':BASE+'tartanground_pilot/exports/tartanground_p2000_k1_results_step200.zip'})

prep=json.loads(d['archives'][BASE+'tartanground_k1_k5_leg_compare/tartanground_k1_k5_leg_compare.zip']['selected_files']['prepared/preparation_metadata.json']['text'])
s += ['## Matched baseline / per-leg K1 / K5 pilot\n',
'**Data:** 28 training trajectories (seven in each environment) and four strict held-out trajectories. Each trajectory contributes two 700-camera-frame windows centered at one-quarter and three-quarters of its available sequence, with boundary clamping and overlap cached once. The retained preparation metadata and freshly counted manifests agree on **28,103 training anchors, 5,632 development anchors and 5,037 strict-test anchors** for each K. The preparation purge is 50 camera frames. A trajectory can occur in both training and development; the strict-test trajectory identities must occur in neither.\n',
table(['Strict held-out trajectory'],[[x] for x in prep['holdout_trajectories']]),
'The pilot evaluation used all 5,037 strict-test anchors, not the later forty-case selection and not the V7 notebook\'s sixteen diagnostic tests. K5 evaluates five horizons at each anchor.\n',
'**Execution:** baseline was assigned physical GPU0 and per-leg physical GPU1 as separate **single-GPU jobs**, not DDP between the two architectures. Each uses batch one, accumulation eight, BF16, TF32, AdamW lr2e-4, warmup10 steps, cosine decay to5% of peak, clip1, weight decay0.05 (separate non-decay group0), seed781. K1 ended at200 optimizer steps; K5 ended at100 optimizer steps. YAML `epochs=100` did not cause 100 epochs: the explicit caps terminated the jobs. K5 initialized model weights from its architecture\'s K1 `best.pt` and began a new optimizer/scheduler.\n',
'**Checkpoint/evaluation:** validation was capped at50 loader batches; `best.pt` minimized composite validation `total`. The strict-test evaluator uses each architecture/K\'s best checkpoint and free rollout. All four jobs reached their recorded cap and the pilot launcher reached `complete` on30 September. The server archive size was12,149,956,904 bytes. This audit read its metadata and selected small members; it did not claim that an older incomplete local12GB transfer was complete.\n']
globals=[]
for arch in ['baseline','per_leg']:
    for k in [1,5]:
        rp=BASE+f'tartanground_k1_k5_leg_compare/{arch}/k{k}'
        g=j(rp+'/strict_test/strict_test_metrics.json')['global']
        globals.append([arch,k,g['q_mae_deg'],g['tau_mae_nm'],g['force_vector_mae'],g['force_vector_rmse'],g['force_direction_cosine_contact'],g['contact_f1']])
s += [table(['Architecture','K','q MAE deg','Torque MAE Nm','Force component MAE N','Force component RMSE N','Contact force cosine','Contact F1'],globals),
'At this capped budget, the added leg transformers increased force error, especially at K5. They slightly increased contact F1. This is the outcome of these trained pilot checkpoints, not proof about their eventual converged ranking. K1 and K5 values have different prediction tasks/horizon aggregation, so comparing their global MAEs alone is not a controlled same-horizon comparison.\n']
selected=['q_mae_deg','tau_mae_nm','force_vector_mae','force_magnitude_mae','force_direction_cosine_contact','contact_f1','load_accuracy']
for arch in ['baseline','per_leg']:
    for k in [1,5]:
        rp=BASE+f'tartanground_k1_k5_leg_compare/{arch}/k{k}'
        mm=j(rp+'/strict_test/strict_test_metrics.json');ms=d['metric_summaries'][rp+'/metrics.jsonl'];cfg=j(rp+'/resolved_config.json')
        ck=meta[rp+'/checkpoints/best.pt']
        label=f'Pilot {arch} K{k}'
        s += [f'### {label}\n',
        f'**Model/training:** {ck["state_numel"]:,} saved parameter scalars; {200 if k==1 else 100} optimizer steps; final logged epoch index{ms["last_train"]["epoch"]}. Best checkpoint step{ck["step"]}; validation total{ck["best_val"]:.12g}. K5 horizon weights are **uniform [0.2,0.2,0.2,0.2,0.2]** in its recorded config. K1 weight is1.\n',
        '**Results — every global exported metric:**\n',table(['Metric','Value'],list(mm['global'].items()))]
        for group in ['by_trajectory','by_horizon','by_leg']:
            s += [f'**{group.replace("_"," ").capitalize()}:**\n',table([group.replace('by_','')]+selected,[[name]+[vals.get(key) for key in selected] for name,vals in mm[group].items()])]
        for group,items in [('global',{'all':mm['global']})]+[(x,mm[x]) for x in ['by_trajectory','by_horizon','by_leg']]:
            for ident,vals in items.items():
                for metric,value in vals.items():detail.append({'experiment':label,'group':group,'group_id':ident,'metric':metric,'value':value})
        ledger.append({'experiment':label,'status':'completed capped pilot and strict evaluation','parameters':ck['state_numel'],'optimizer_steps':ck['step'],'epoch_note':f'epoch index {ck["epoch"]};cap overrides100configuredepochs','test_scope':'5037unseenanchors;4trajectories','best_val_total':ck['best_val'],'source':rp})

s += ['### Development-only horizon-weight calibration\n',
'Three K5 candidates each trained20 optimizer steps on the same calibration setup and were scored on development data. Candidate selection minimized the normalized physical score described below; strict-test data did not select the weights. Each candidate starts from the corresponding K1 initialization rather than continuing another candidate.\n']
for stage in ['tartanground_k1_k5_leg_compare','tartanground_k1_k5_leg_compare_10ep_test10']:
    cal=j(BASE+stage+'/lambda_calibration/lambda_calibration.json')
    s += [f'**{stage}:** selected `{cal["selected"]}`.\n',table(['Candidate','Horizon weights','Development physical score'],[[n,str(v['weights']),v['development_physical_score']] for n,v in cal['candidates'].items()])]
    for n,v in cal['candidates'].items():ledger.append({'experiment':stage+' calibration '+n,'status':'completed20-stepdevcalibration','parameters':110526384,'optimizer_steps':20,'epoch_note':'cappedcalibration','test_scope':'development only','best_val_total':None,'source':BASE+stage+'/lambda_calibration/'+n})
s += ['Thus the earlier pilot used uniform weights; the ten-epoch study used strong decay `[0.516,0.258,0.129,0.065,0.032]`. Source equivalence does not imply identical configuration or identical checkpoints.\n']

s += ['## Ten-epoch comparison with ten selected cases per unseen trajectory\n',
'**Recipe:** same prepared training/development manifests, same parameterized architectures and separate GPU assignments, `epochs=10`, `max_steps=null`, batch1/accumulation8, lr2e-4, AdamW settings as above. K1 validates/checkpoints every50 optimizer steps; K5 every25. Ten uninterrupted passes imply35,130 optimizer steps. K5 initializes only the model weights from its architecture\'s ten-epoch K1 best checkpoint. Teacher forcing is configured on, with about half of training rollout steps forced by the mask; evaluation is free rollout.\n',
'**Resume semantics:** the trainer saves epoch and global optimizer step but no batch cursor. A mid-epoch resume begins that epoch\'s loader again. The K1 log contains a failed RNG restoration attempt followed by a successful resume at baseline epoch4/step15,600 and per-leg epoch3/step11,950. Consequently the completed K1 runs have extra updates: **1,548 baseline**, **1,411 per-leg** beyond35,130. This is why their final steps exceed the uninterrupted ten-epoch total. Epoch index9 is the tenth epoch; a completed latest checkpoint stores epoch10.\n']
rows=[]
for arch in ['baseline','per_leg']:
    for k in [1,5]:
        rp=BASE+f'tartanground_k1_k5_leg_compare_10ep_test10/{arch}/k{k}'
        ms=d['metric_summaries'][rp+'/metrics.jsonl'];latest=meta[rp+'/checkpoints/latest.pt'];best=meta[rp+'/checkpoints/best.pt'];v=ms['last_validation']
        completed=latest['epoch']==10
        status='Completed training' if completed else 'Paused in tenth epoch'
        rows.append([arch,k,status,ms['last_train']['step'],latest['step'],best['step'],best['best_val'],v['total']])
        ledger.append({'experiment':f'Tenepoch {arch} K{k}','status':status,'parameters':latest['state_numel'],'optimizer_steps':ms['last_train']['step'],'epoch_note':f'latest checkpoint epoch{latest["epoch"]},loggedepoch{ms["last_train"]["epoch"]}','test_scope':'40selectedunseenanchors;finalevaluationnotproduced','best_val_total':best['best_val'],'source':rp})
s += [table(['Architecture','K','Status','Last logged step','Latest durable step','Best step','Best dev total','Latest dev total'],rows),
'**Per-leg K5:** paused at the audited process snapshot; last logged step33,477, durable checkpoint33,475, leaving1,653 nominal optimizer steps from the logged counter. This is roughly4.7% of nominal K5 updates. That fraction is not an ETA while the process is suspended. Its latest validation total5713.12464844 is far above its best8.97188745618 at step19,300. The late-run objective degraded; the best checkpoint remains the earlier one. The source normalization/FK issues and exposure change identified in the architecture audit are relevant limits, but the retained logs do not establish a unique cause of this degradation.\n',
'**Strict test selection:** exactly40 shared anchors, ten evenly spaced common valid camera anchors for each of the four held-out trajectories. The selected set is the same for K1 and K5 and both architectures. Forty cases means40 input anchors for each model/K evaluation. With four legs, this yields160 leg rows for K1 and800 for K5, or1,920 rows across two architectures and both K values. These row counts are derived packaging dimensions, not1,920 independent trajectory cases.\n',
'**Results status:** the launcher remains in `k5_baseline_and_per_leg`. It has not reached its recorded strict-evaluation/package phases. No ten-epoch strict-test metrics or completed `tartanground_10epoch_unseen_test_only.zip` were found in that run\'s collected inventory. The pilot metrics above must not be relabeled as ten-epoch results. The test-only packager is implemented to require the forty-case metrics/prediction/selection/CSV artifacts for all four components and excludes training checkpoints, logs, source and raw caches. The existing `tartanground_10ep_update.zip` is a source/configuration update bundle, not a result archive.\n']
for arch in ['baseline','per_leg']:
    for k in [1,5]:
        rp=BASE+f'tartanground_k1_k5_leg_compare_10ep_test10/{arch}/k{k}'
        ms=d['metric_summaries'][rp+'/metrics.jsonl']
        s += [f'### Ten-epoch {arch} K{k} recorded development losses\n',
        'Best and latest below are retained development events, each capped to50 loader batches. They have not been converted to physical strict-test metrics. The best event is selected by `total`, not force MAE alone.\n',
        table(['Loss/score','Best-total event','Latest event'],[[key,ms['best_validation'].get(key),ms['last_validation'].get(key)] for key in ms['last_validation'] if key!='type'])]

s += ['## Isolated V7-F-only continuation: implemented, queued\n',
'The isolated continuation source and YAML are present. It freezes the merged world/dynamics trunk and trains only the2,448,554-parameter V7-F specialist, with full-development normalized force MAE selecting checkpoints, scheduled teacher forcing0.5→0 over seven epochs, and full strict-manifest evaluation. The retained phase file reads **`waiting_for_comparison_pid_284634`** at2 October05:45:40IST. No completed continuation training/result record was collected. This is a configured queued experiment; it has no measured result to place next to the pilot. Its exact proposed recipe and objective are in the architecture section/configuration appendix.\n']
ledger.append({'experiment':'IsolatedV7Fforcecontinuation','status':'queued waiting for comparison','parameters':2448554,'optimizer_steps':None,'epoch_note':'not executed in collected evidence','test_scope':'planned fullstrictmanifest','best_val_total':None,'source':'/path/to/ugp/merged_feelwm_v7_force_finetune'})

(root/'report_feelwm_runs.md').write_text('\n'.join(s),encoding='utf-8')
for name,items in [('UGP_RTX_run_ledger.csv',ledger),('UGP_TartanGround_all_strict_metrics.csv',detail)]:
    fields=list(dict.fromkeys(k for row in items for k in row))
    with (out/name).open('w',newline='',encoding='utf-8-sig') as fp:
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(items)
print(json.dumps({'section_bytes':(root/'report_feelwm_runs.md').stat().st_size,'ledger_rows':len(ledger),'strict_metric_rows':len(detail)}))
