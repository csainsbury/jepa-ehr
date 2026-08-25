#!/usr/bin/env python3
"""Fable-corrected 3x3 generator/model restoration of the fixed L1 recipe beta."""
from __future__ import annotations
import json,torch
from clinical_jepa.eval.j04c_falsifier import CAL_OOD,PROBE_FIT,TRAIN,fit_stage0_time_transform,generate_factor_split
from clinical_jepa.eval.j04c_initialization_bridge import train_recipe_decoupled_seeds
from clinical_jepa.eval.j04c_nuisance_bridge import independent_train_nuisance
from clinical_jepa.eval.j04c_stage1 import evaluate_readouts,fit_condition_readouts,freeze_encoder,threshold_reference_report,trained_collapse_diagnostics
GENERATORS=(1101,1102,1103);MODELS=(2101,2102,2103);READOUT=2102;RECIPE='L1_AVG';WEIGHT=5.;FLOOR=.01

def main():
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1);reference,digest=threshold_reference_report();results={};matrices=[[[] for _ in GENERATORS] for _ in range(3)]
 for gi,g in enumerate(GENERATORS):
  train=independent_train_nuisance(generate_factor_split(g,TRAIN,8192),g);probe=generate_factor_split(g,PROBE_FIT,2048);cal=generate_factor_split(g,CAL_OOD,2048);transform=fit_stage0_time_transform(train)
  for seed in MODELS:
   c=train_recipe_decoupled_seeds(train,transform,RECIPE,seed,seed,schedule_seed=seed,identity_predictor=True,total_steps=2000,directional_variance_weight=WEIGHT,directional_variance_floor=FLOOR,per_identity_directional_hinge=True);freeze_encoder(c);probes,head,_=fit_condition_readouts(c,probe,transform,READOUT);ev,_=evaluate_readouts(c,probes,head,cal,transform);ba=[f['balanced_accuracy'] for f in ev['factors']];rows,position=trained_collapse_diagnostics(c,cal,transform,reference)
   for factor,value in enumerate(ba):matrices[factor][gi].append(value)
   results[f'g{g}_m{seed}']={'factor_ba':ba,'constraint_transfer_all_pass':all(r['both_metrics_pass'] for r in rows),'constraint_transfer_rows':rows,'position_specificity_descriptive':position,'last_100_train_min_identity_directional_variance':c.training['losses']['components']['v_direction_min']['last_100_mean']}
 within_ranges=[[max(row)-min(row) for row in matrices[f]] for f in range(3)];global_ranges=[max(x for row in matrices[f] for x in row)-min(x for row in matrices[f] for x in row) for f in range(3)];checks={'constraint_transfer_all_four_identities_all_nine_cells':all(v['constraint_transfer_all_pass'] for v in results.values()),'all_factor_ba_at_least_0_80':min(x for factor in matrices for row in factor for x in row)>=.8,'all_within_generator_model_ranges_at_most_0_05':max(x for factor in within_ranges for x in factor)<=.05}
 print(json.dumps({'schema':'BP011-J04C-L1-BETA-3X3-V1','contract':{'axis':'restore generator variation around the fixed-generator L1 recipe beta after the untrained baseline cleared the frozen learned-signal rule','constraint_transfer':'TRAIN-side per-identity floor must hold OOD variance above unchanged identity thresholds; not independent anti-collapse confirmation','pass':'all four identity constraints pass in 9/9 cells; all factor BA >=0.80; within-generator model-seed ranges <=0.05','descriptive':'global factor ranges, last-100 TRAIN minimum identity variance, and position specificity','no_tuning_after_open':True},'stage0_untrained_contrast':{'trained_minus_untrained':[0.06615171415524768,0.06771511880389591,0.1039253186136272],'threshold':0.05,'all_factors_pass':True},'generator_seeds':GENERATORS,'model_seeds':MODELS,'common_readout_seed':READOUT,'recipe':RECIPE,'directional_variance_weight':WEIGHT,'directional_variance_floor':FLOOR,'per_identity_directional_hinge':True,'factor_ba_matrices_rows_generator_cols_model':matrices,'within_generator_model_seed_ranges':within_ranges,'global_factor_ranges':global_ranges,'checks':checks,'contract_pass':all(checks.values()),'results':results,'threshold_digest':digest,'claim_ceiling':'FIXED_RECIPE_PUBLIC_SYNTHETIC_L1_VIABILITY_ACROSS_GENERATOR_AND_MODEL_SEEDS'},sort_keys=True,separators=(',',':'),allow_nan=False))
if __name__=='__main__':main()
