import csv, hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

root=Path(__file__).parent
base=root/'server_snapshot/path/to/ugp/V7F_4080_Run/storage/V8_contactmap'
def load(path):
    with (base/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def numbers(rows,key):return [float(r[key]) for r in rows if r[key]!='']
def mean(rows,key):return statistics.mean(numbers(rows,key))
def group(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[r[key]].append(r)
    return groups
out={}
a=load('A_memory/V8A_CAUSAL_MEMORY_CONTACT_COVERAGE_RAW.csv')
out['A']={'queries':len(a),'hits':sum(r['found']=='True' for r in a),'hit_rate':sum(r['found']=='True' for r in a)/len(a),'median_hit_distance_m':statistics.median(numbers(a,'distance_m')),'median_hit_age_s':statistics.median(numbers(a,'age_s')),'seen_rates':{k:sum(r[k]=='True' for r in a)/len(a) for k in ['seen_0p2s_earlier','seen_0p5s_earlier','seen_1p0s_earlier','seen_2p0s_earlier']}}
b=load('B_touchdown/V8B_16TEST_TOUCHDOWN_RAW.csv')
timing=load('B_touchdown/V8B_CONTACT_TIMING.csv')
out['B']={'rows':len(b),'contact_positive_predictions':sum(float(r['contact_prob'])>=.5 for r in b),'true_contact_rows':sum(float(r['contact_true'])>=.5 for r in b),'contact_prob_range':[min(numbers(b,'contact_prob')),max(numbers(b,'contact_prob'))],'timing_rows':len(timing),'finite_timing_rows':len(numbers(timing,'timing_abs_error_s')),'median_timing_abs_s':statistics.median(numbers(timing,'timing_abs_error_s')),'mean_timing_abs_s':mean(timing,'timing_abs_error_s'),'max_timing_abs_s':max(numbers(timing,'timing_abs_error_s')),'timing_error_counts':dict(Counter(r['timing_abs_error_s'] for r in timing)),'gt_first_counts':dict(Counter(r['gt_first_contact_s'] for r in timing)),'pred_first_counts':dict(Counter(r['pred_first_contact_s'] for r in timing))}
c=load('C_contact_field/V8C_16TEST_CONTACT_FIELD_RAW.csv')
out['C']={'rows':len(c),'columns':list(c[0]),'head':c[:2],'terrain_rows':dict(Counter(r['terrain_type'] for r in c)),'all_metrics':{k:mean(c,k) for k in ['tau_MAE','force_MAE','force_vector_error','force_sigma_mean']}}
d=load('D_counterfactual/V8D_MATCHED_PAIR_RAW.csv')
out['D']={'rows':len(d),'columns':list(d[0]),'head':d[:2]}
e=load('E_microplanner/V8E_16TEST_SHADOW_PLANS.csv')
ec=load('E_microplanner/V8E_ALL_CANDIDATES_TOPK.csv')
def estats(rows):return {'plans':len(rows),'active_leg_counts':dict(Counter(r['active_legs'] for r in rows)),'mean_predicted_risk_reduction':mean(rows,'predicted_risk_reduction'),'median_predicted_risk_reduction':statistics.median(numbers(rows,'predicted_risk_reduction')),'positive_risk_reduction_plans':sum(float(r['predicted_risk_reduction'])>0 for r in rows),'mean_support_area_m2':mean(rows,'support_area_m2'),'mean_joint_score':mean(rows,'joint_score'),'mean_nominal_independent_score':mean(rows,'nominal_independent_score')}
out['E']={'all':estats(e),'by_terrain':{k:estats(v) for k,v in group(e,'terrain_type').items()},'candidates':len(ec),'candidate_groups':len(set((r['terrain_type'],r['test_index'],r['leg']) for r in ec)),'candidate_offset_range_m':[min(numbers(ec,'offset_m')),max(numbers(ec,'offset_m'))],'candidate_offset_over_0p22':sum(float(r['offset_m'])>.22 for r in ec),'columns':list(ec[0]),'candidate_horizons':dict(Counter(r['horizon_s'] for r in ec))}
f=load('F_active_adaptation/V8F_CAUSAL_CONTACT_REPLAY_RAW.csv')
def fstats(rows):
    metrics={k:mean(rows,k) for k in ['base_force_MAE','adapt_force_MAE','base_tau_MAE','adapt_tau_MAE']}
    metrics.update({'samples':len(rows),'force_improvement_rate':sum(r['improved_force']=='True' for r in rows)/len(rows),'tau_improvement_rate':sum(r['improved_tau']=='True' for r in rows)/len(rows),'force_worsened':sum(float(r['adapt_force_MAE'])>float(r['base_force_MAE']) for r in rows),'force_equal':sum(float(r['adapt_force_MAE'])==float(r['base_force_MAE']) for r in rows),'tau_worsened':sum(float(r['adapt_tau_MAE'])>float(r['base_tau_MAE']) for r in rows),'tau_equal':sum(float(r['adapt_tau_MAE'])==float(r['base_tau_MAE']) for r in rows)})
    metrics['force_improvement_percent']=100*(metrics['base_force_MAE']-metrics['adapt_force_MAE'])/metrics['base_force_MAE']
    metrics['tau_improvement_percent']=100*(metrics['base_tau_MAE']-metrics['adapt_tau_MAE'])/metrics['base_tau_MAE']
    return metrics
voi=load('F_active_adaptation/V8F_SAFE_CURIOSITY_SHADOW.csv')
out['F']={'all':fstats(f),'by_terrain':{k:fstats(v) for k,v in group(f,'terrain_type').items()},'curiosity_groups':len(voi),'curiosity_changed':sum(r['same_as_safest']=='False' for r in voi),'mean_extra_information':statistics.mean(float(r['active_info_gain'])-float(r['safest_info_gain']) for r in voi),'mean_extra_score':statistics.mean(float(r['active_score'])-float(r['safest_score']) for r in voi),'curiosity_changes':[r for r in voi if r['same_as_safest']=='False']}
manifest=load('V8_SHA256_MANIFEST.csv')
checks=[]
for r in manifest:
    p=base/r['file']
    if p.exists():checks.append({'file':r['file'],'matches':hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],'size_matches':p.stat().st_size==int(r['size_bytes'])})
out['manifest']={'rows':len(manifest),'local_available':len(checks),'checks':checks,'not_copied':[r['file'] for r in manifest if not (base/r['file']).exists()]}
linear=lambda x,y:x*y+y
ln=lambda x:2*x
b_counts={'input':linear(30,128),'leg_identity':4*128,'position_embedding':128*128,'3_transformer_layers':3*((3*128*128+3*128)+linear(128,128)+linear(128,512)+linear(512,128)+2*ln(128)),'final_norm':ln(128),'horizon_projector':linear(1,64)+linear(64,128)+ln(128),'fusion':ln(256)+linear(256,256)+linear(256,128),'output_heads':linear(128,1)+2*linear(128,3)}
c_counts={'trunk':ln(297)+linear(297,512)+linear(512,256)+ln(256),'gate':linear(256,128)+linear(128,4),'4_experts':4*(linear(256,256)+linear(256,12))}
d_counts={'input_norm':ln(1321),'mlp':linear(1321,768)+linear(768,384)+linear(384,192),'output':linear(192,12)}
out['source_derived_params']={k:{'parts':v,'total':sum(v.values())} for k,v in [('B',b_counts),('C',c_counts),('D',d_counts)]}
(root/'v8_rtx_aggregates.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
