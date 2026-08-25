#!/usr/bin/env python3
"""Keep L2 target beta at 0.1 while restoring the unscaled L2 constraint strength."""
from __future__ import annotations
import json,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_l1_plus_l2_beta
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
G=1102;MODELS=(2101,2102,2103);READOUT=2102;BETA=.1;UNTRAINED=(0.7753685818549992,0.7805427899230244,0.75101867607262)

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);train=independent_train_nuisance(generate_factor_split(G,TRAIN,8192),G);probe=generate_factor_split(G,PROBE_FIT,2048);cal=generate_factor_split(G,CAL_OOD,2048);transform=fit_stage0_time_transform(train);reference,digest=threshold_reference_report();results={};values=[[],[],[]]
 for seed in MODELS:
  c=train_l1_plus_l2_beta(train,transform,seed,l2_beta=BETA,total_steps=2000,scale_l2_constraint_with_beta=False);freeze_encoder(c);probes,head,_=fit_condition_readouts(c,probe,transform,READOUT);ev,_=evaluate_readouts(c,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=trained_collapse_diagnostics(c,cal,transform,reference)
  for i,v in enumerate(ba):values[i].append(v)
  results[str(seed)]={'factor_ba':ba,'l2_constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'l2_constraint_transfer_rows':rows,'l2_position_specificity_descriptive':position,'training_loss_summary':c.training['losses']}
 means=[sum(v)/3 for v in values];deltas=[a-b for a,b in zip(means,UNTRAINED)];ranges=[max(v)-min(v) for v in values];checks={'all_sixteen_l2_constraints_all_three_seeds':all(v['l2_constraint_transfer_all_pass'] for v in results.values()),'all_factor_ba_at_least_0_80':min(x for row in values for x in row)>=.8,'each_factor_seed_range_at_most_0_05':max(ranges)<=.05,'trained_minus_untrained_mean_at_least_0_05_each_factor':min(deltas)>=.05}
 print(json.dumps({'schema':'BP011-J04C-L1-ANCHORED-L2-UNSCALED-CONSTRAINT-V1','contract':{'axis':'retain full four-layer L2 target beta 0.1 but stop beta-scaling its independent directional constraint after a 47/48 constraint pass','genuine_l2':'all 16 separated targets contribute nonzero cosine and directional-constraint gradients','pass':'all 16 L2 constraints pass in 3/3; all BA >=0.80; factor ranges <=0.05; mean trained-minus-untrained >=0.05 per factor','no_tuning_after_open':True},'generator_seed':G,'model_seeds':MODELS,'common_readout_seed':READOUT,'recipe':'L1_AVG+0.1*L2_SEP','l2_beta':BETA,'l1_directional_weight':5.0,'l2_directional_weight':20.0,'directional_floor':0.01,'scale_l2_constraint_with_beta':False,'trained_factor_ba_by_factor':values,'trained_factor_ba_means':means,'trained_minus_untrained_mean_ba':deltas,'factor_seed_ranges':ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_GENERATOR_PUBLIC_SYNTHETIC_L1_ANCHORED_NONZERO_L2_BETA'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
