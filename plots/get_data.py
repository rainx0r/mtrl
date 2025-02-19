import wandb
from wandb.apis.public import Run
import json
from collections import defaultdict

api = wandb.Api()
runs = api.runs('reggies-phd-research/mtrl-mt50-results')


# run_names = ['mt10_mtmhsac_task_weights_false', 'mt10_softmodules_task_weights_false', 'mt10_moore_fix', 'mt10_mtmhsac_sm_params_3_layers_v2', 'mt10_mtmhsac_v2_moore_params_3_layers',
#     'mt10_mtmhsac_paco_params_v2_3_layers', 'mt10_paco', 'mt10_mtmhsac_v2_width_1024', 'mt10_mtmhsac_v2_width_2048', 'mt10_mtmhsac_v2_width_4096']


run_names = ['mt50_mtmhsac_v2', 'mt50_softmodules_task_weights_false', 'mt50_moore', 'mt50_mtmhsac_moore_params_v2', 
 'mtmhsac_moore_params_log_std-10_clipped_q', 'mt50_paco', 'mt50_mtmhsac_sm_params_v2', 
 'mt50_mtmhsac_v2_paco_params_3_layers', 'mt50_mtmhsac_v2_2048_width', 'mt50_mtmhsac_v2_4096_width']

#run_names = ['mt10_paco', 'mt10_softmodules_task_weights_false', 'mt10_moore_fix', 'mt10_mtmhsac_task_weights_false', 'mt10_care']


mt10_results = defaultdict(list)

for run in runs:
    if run.name in run_names and run.state != 'running':
        mt10_results[run.name].append(run.summary['charts/mean_success_rate'])

print(mt10_results)
