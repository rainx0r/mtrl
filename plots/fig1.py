import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats

def interquartile_mean(data):
    q1, q3 = np.percentile(data, [25, 75])
    mask = (data >= q1) & (data <= q3)
    return np.mean(data[mask])

def plot_results(mt10_results, mt50_results, param_counts_mt10, param_counts_mt50, mtrl_archs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mt10_results)))
    
    # Plot MT10 results
    mtrl = []
    mtrl_i = []
    ffnn = []
    ffnn_i = []

    for i, (name, data) in enumerate(mt10_results.items()):
        data = np.array(data)
        # x = np.full_like(data, param_counts_mt10[i])
        # ax1.scatter(param_counts_mt10[i], np.mean(data), color=colors[i], alpha=0.6)
        iqm = scipy.stats.trim_mean(data, proportiontocut=0.25, axis=None) # interquartile_mean(data)
        if name in mtrl_archs:
            mtrl.append(iqm)
            mtrl_i.append(i)
            ax1.plot(i, iqm)
            # ax1.scatter(i, iqm, color='green', alpha=0.6, marker='o') # width=1, label=name, edgecolor = "black", yerr=np.std(data)/np.sqrt(len(data)))
        else:
            name = name.replace('Params', '')
            ffnn.append(iqm)
            ffnn_i.append(i)
            # ax1.scatter(i, iqm, color='orange', alpha=0.6, marker='X') # , width=1, label=name, edgecolor = "black", yerr=np.std(data)/np.sqrt(len(data)))
        # ax1.text(i, 0.605, name, ha='center', va='bottom', color='black', fontsize=16, rotation=45)
    print(mtrl_i, mtrl)
    print(ffnn, ffnn_i)
    # ax1.plot(mtrl_i, mtrl)
    ax1.plot(ffnn, ffnn_i)

    ax1.set_ylim(0.6, 1.0)
    ax1.set_xticks(range(len(list(param_counts_mt10))))
    ax1.set_xticklabels([f'{p/1e6:.1f}M' for p in list(param_counts_mt10)], fontsize='x-large')

    ax1.set_ylabel('Success %', fontsize='x-large')
    ax1.set_xlabel('# of Parameters', fontsize='x-large')
    legend_elements = [
       Patch(facecolor='green', label='MTRL-Specific Architecture'),
       Patch(facecolor='orange', label='Simple Feed-Forward')
    ]

    ax1.legend(handles=legend_elements, loc='upper left', fontsize='x-large')


    for i, (name, data) in enumerate(mt50_results.items()):
        data = np.array(data)
        # x = np.full_like(data, param_counts_mt50[i])
        # ax1.scatter(param_counts_mt50[i], np.mean(data), color=colors[i], alpha=0.6)
        iqm = interquartile_mean(data)
        if name in mtrl_archs:
            ax2.bar(i, iqm, color='green', alpha=0.6, width=1, label=name, edgecolor = "black", yerr=np.std(data)/np.sqrt(len(data)))
        else:
            name = name.replace('Params', '')
            ax2.bar(i, iqm, color='orange', alpha=0.6, width=1, label=name, edgecolor = "black", yerr=np.std(data)/np.sqrt(len(data)))
        ax2.text(i, 0.305, name, ha='center', va='bottom', color='black', fontsize=12, rotation=45)


    ax2.set_ylim(0.3, 1.0)
    ax2.set_xticks(range(len(list(param_counts_mt50))))
    ax2.set_xticklabels([f'{p/1e6:.1f}M' for p in list(param_counts_mt50)])

    ax2.set_ylabel('Success %', fontsize='x-large')
    ax2.set_xlabel('# of Parameters', fontsize='x-large')

    ax1.set_title('MT10 Benchmark')
    ax2.set_title('MT50 Benchmark')
    plt.tight_layout()
    return fig



def plot_learning_curves(simple_results, mtrl_results, params_dict, mt50_simple, mt50_mtrl, params_dict_mt50):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))    # plt.figure(figsize=(10, 6))
    
    # Set up x-axis parameter counts
    param_counts = sorted(list(simple_results.keys()) + [params for params, _, _ in mtrl_results.values()])
    
    # Plot simple architectures
    simple_params = sorted(simple_results.keys())
    values = [simple_results[p] for p in simple_params]
    errors = [simple_results_std[p] for p in simple_params]
    
    ax1.plot(list(params_dict.values()), values, '-o', color='blue', label='Simple FF')
    ax1.fill_between(list(params_dict.values()), 
                    np.array(values) - np.array(errors),
                    np.array(values) + np.array(errors),
                    alpha=0.2, color='blue')
    
    # Plot MTRL architectures as separate points
    for name, (params, value, error) in mtrl_results.items():
        ax1.scatter(params_dict[params], value, label=name, marker='*', s=100)
        ax1.errorbar(params_dict[params], value, yerr=error, fmt='none', capsize=5)
        print(params_dict[params], params)

    
    # Format x-axis with parameter counts
    #ax1.xscale('log')
    # ax1.xticks(param_counts, [f'{p/1e6:.1f}M' for p in param_counts], rotation=45)
    ax1.set_ylim(0.6, 1.0)
    ax1.set_xticks(range(len(params_dict)))
    ax1.set_xticklabels([f'{p/1e6:.1f}M' for p in list(params_dict.keys())], fontsize='x-large')

    ax1.set_ylabel('Success %', fontsize='x-large')
    ax1.set_xlabel('# of Parameters', fontsize='x-large')
    #legend_elements = [
    #   Patch(facecolor='green', label='MTRL-Specific Architecture'),
    #   Patch(facecolor='orange', label='Simple Feed-Forward')
    #]

    #ax1.legend(handles=legend_elements, loc='upper left', fontsize='x-large')    
    #ax1.xlabel('Number of Parameters')
    #ax1.ylabel('Success Rate')
    #ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize='x-large')
    
    param_counts = sorted(list(mt50_simple.keys()) + [params for params, _, _ in mt50_mtrl.values()])

    # Plot simple architectures
    simple_params = sorted(mt50_simple.keys())

    simple_results_std_mt50 = {
    param_counts_mt50[0]: np.std(mt50['SAC']),
    param_counts_mt50[1]: np.std(mt50['SM Params']),
    param_counts_mt50[2]: np.std(mt50['MOORE Params']),
    param_counts_mt50[3]: np.std(mt50['PaCo Params']),
    param_counts_mt50[4]: np.std(mt50['2048']),
    param_counts_mt50[5]: np.std(mt50['4096'])
    }

    print(simple_results, simple_params)


    values = [mt50_simple[p] for p in simple_params]
    errors = [simple_results_std_mt50[p] for p in simple_params]

    ax2.plot(list(params_dict_mt50.values()), values, '-o', color='blue', label='Simple FF')
    ax2.fill_between(list(params_dict_mt50.values()),
                    np.array(values) - np.array(errors),
                    np.array(values) + np.array(errors),
                    alpha=0.2, color='blue')

    # Plot MTRL architectures as separate points
    for name, (params, value, error) in mt50_mtrl.items():
        ax2.scatter(params_dict_mt50[params], value, label=name, marker='*', s=100)
        ax2.errorbar(params_dict_mt50[params], value, yerr=error, fmt='none', capsize=5)

    ax2.set_xticks(range(len(params_dict_mt50)))
    ax2.set_xticklabels([f'{p/1e6:.1f}M' for p in list(params_dict_mt50.keys())], fontsize='x-large')

    ax2.set_ylabel('Success %', fontsize='x-large')
    ax2.set_xlabel('# of Parameters', fontsize='x-large')
    
    ax2.legend(loc='upper left', fontsize='x-large')

    ax1.set_title('MT10 Benchmark')
    ax2.set_title('MT50 Benchmark')
    plt.tight_layout()

    return fig


# Test data
param_counts_mt10 = np.array([1700000, 370000, 690000, 1380000, 690000, 1380000, 1700000, 8660000, 2200000, 34000000])
param_counts_mt50 = np.array([2e4, 5e4, 1e5, 3e5, 5e5, 1e6, 3e6, 5e6, 8e6, 1e7])

mt10_results = {'PaCo': [0.8320000000000001, 0.736, 0.5900000000000001, 0.72, 0.778, 0.756, 0.748, 0.728, 0.738, 0.6], 
'MTMHSAC': [0.884, 0.868, 0.72, 0.728, 0.752, 0.756, 0.768, 0.756, 0.742, 0.752], 
'SM': [0.744, 0.858, 0.778, 0.71, 0.942, 0.772, 0.95, 0.93, 0.764, 0.772], 
'MOORE': [0.842, 0.778, 0.788, 0.968, 0.788, 0.868, 0.7939999999999999, 0.766, 0.774, 0.9760000000000002], 
'SM Params': [0.8460000000000001, 0.8560000000000001, 0.8699999999999999, 0.868, 0.8400000000000001, 0.8620000000000001, 0.8699999999999999, 0.8300000000000001, 0.842, 0.868], 
'MOORE Params': [0.8800000000000001, 0.8800000000000001, 0.8780000000000001, 0.8800000000000001, 0.8780000000000001, 0.8539999999999999, 0.8400000000000001, 0.866, 0.8699999999999999, 0.868], 
'PaCo Params': [0.876, 0.932, 0.8699999999999999, 0.8620000000000001, 0.9280000000000002, 0.8719999999999999, 0.86, 0.868, 0.85, 0.868], 
'Width 2048': [0.952, 0.932, 0.964, 0.932, 0.89, 0.844, 0.918, 0.94, 0.86, 0.918], 
'Width 1024': [0.844, 0.8480000000000001, 0.8560000000000001, 0.8879999999999999, 0.884, 0.866, 0.8800000000000001, 0.9399999999999998, 0.892, 0.8460000000000001], 
'Width 4096': [0.954, 0.968, 0.958, 0.85, 0.966, 0.92, 0.908, 0.876, 0.936, 0.948]}

mt10 = {}
param_counts_mt10 = np.array([370000, 690000, 690000, 1380000, 1380000, 1700000, 1700000, 2200000, 8660000, 34000000])
mt10['SAC'] = mt10_results['MTMHSAC']
mt10['SM'] = mt10_results['SM']
mt10['SM Params'] = mt10_results['SM Params']
mt10['MOORE'] = mt10_results['MOORE']
mt10['MOORE Params'] = mt10_results['MOORE Params']
mt10['PaCo'] = mt10_results['PaCo']
mt10['PaCo Params'] = mt10_results['PaCo Params']
mt10['1024'] = mt10_results['Width 1024']
mt10['2048'] = mt10_results['Width 2048']
mt10['4096'] = mt10_results['Width 4096']


mt50_results = {'mt50_mtmhsac_v2': [0.524, 0.54, 0.014000000000000002, 0.5472, 0.5472], 
'mt50_softmodules_task_weights_false': [0.638, 0.6688, 0.5924, 0.3972, 0.5328], 
'mt50_paco': [0.6888, 0.5864, 0.5892000000000001, 0.5568000000000001, 0.5808], 
'mt50_moore': [0.6507999999999999, 0.6476000000000001, 0.6576, 0.6531999999999999, 0.6324], 
'mt50_mtmhsac_moore_params_v2': [0.5544, 0.5096, 0.47839999999999994, 0.01, 0.0168], 
'mtmhsac_moore_params_log_std-10_clipped_q': [0.736, 0.7203999999999999, 0.7556, 0.7228, 0.7584000000000001], 
'mt50_mtmhsac_sm_params_v2': [0.6831999999999999, 0.682, 0.6476000000000001, 0.6752000000000001, 0.6607999999999999], 
'mt50_mtmhsac_v2_paco_params_3_layers': [0.8048000000000001, 0.7996, 0.778, 0.8284, 0.364], 
'mt50_mtmhsac_v2_2048_width': [0.7071999999999999, 0.7191999999999998, 0.6095999999999999, 0.7680000000000001, 0.8623999999999999], 
'mt50_mtmhsac_v2_4096_width': [0.2304, 0.4503999999999999, 0.5851999999999999, 0.5472, 0.5791999999999999]}

mt50_results = {'mt50_mtmhsac_v2': [0.524, 0.54, 0.014000000000000002, 0.5472, 0.5472], 'mt50_softmodules_task_weights_false': [0.638, 0.6688, 0.5924, 0.3972, 0.5328], 'mt50_paco': [0.6888, 0.5864, 0.5943999999999999, 0.5568000000000001, 0.5808], 'mt50_moore': [0.6507999999999999, 0.6476000000000001, 0.6576, 0.6531999999999999, 0.6324], 'mt50_mtmhsac_moore_params_v2': [0.5544, 0.5096, 0.47839999999999994, 0.01, 0.0168], 'mtmhsac_moore_params_log_std-10_clipped_q': [0.736, 0.7203999999999999, 0.7556, 0.7228, 0.7584000000000001], 'mt50_mtmhsac_sm_params_v2': [0.6831999999999999, 0.682, 0.6476000000000001, 0.6752000000000001, 0.6607999999999999], 'mt50_mtmhsac_v2_paco_params_3_layers': [0.8048000000000001, 0.7996, 0.778, 0.7831999999999999, 0.8351999999999999], 'mt50_mtmhsac_v2_2048_width': [0.7816, 0.8708000000000001, 0.8, 0.7456, 0.8608], 'mt50_mtmhsac_v2_4096_width': [0.7611999999999999, 0.8640000000000001, 0.8032, 0.7980000000000002]}

mt50 = {}
param_counts_mt50 = np.array([517200, 1606140, 1606140, 2181500, 2181500, 6801180, 6801180, 9396624, 35570064])
mt50['SAC'] = mt50_results['mt50_mtmhsac_v2']
mt50['SM'] = mt50_results['mt50_softmodules_task_weights_false']
mt50['SM Params'] = mt50_results['mt50_mtmhsac_sm_params_v2']
mt50['MOORE'] = mt50_results['mt50_moore']
mt50['MOORE Params'] = mt50_results['mtmhsac_moore_params_log_std-10_clipped_q']
mt50['PaCo'] = mt50_results['mt50_paco']
mt50['PaCo Params'] = mt50_results['mt50_mtmhsac_v2_paco_params_3_layers']
mt50['2048'] = mt50_results['mt50_mtmhsac_v2_2048_width']
mt50['4096'] = mt50_results['mt50_mtmhsac_v2_4096_width']

mtrl_archs = ['SM', 'MOORE', 'PaCo']

mtrl_results = {
    'SM': (param_counts_mt10[1], scipy.stats.trim_mean(mt10['SM'], proportiontocut=0.25, axis=None), np.std(mt10['SM'])),
    'MOORE': (param_counts_mt10[3], scipy.stats.trim_mean(mt10['MOORE'], proportiontocut=0.25, axis=None), np.std(mt10['MOORE'])),
    'PaCo': (param_counts_mt10[5], scipy.stats.trim_mean(mt10['PaCo'], proportiontocut=0.25, axis=None), np.std(mt10['PaCo']))
}

# Set up simple scaled architectures
simple_results = {
    param_counts_mt10[0]: scipy.stats.trim_mean(mt10['SAC'], proportiontocut=0.25, axis=None),
    param_counts_mt10[1]: scipy.stats.trim_mean(mt10['SM Params'], proportiontocut=0.25, axis=None),
    param_counts_mt10[3]: scipy.stats.trim_mean(mt10['MOORE Params'], proportiontocut=0.25, axis=None),
    param_counts_mt10[5]: scipy.stats.trim_mean(mt10['PaCo Params'], proportiontocut=0.25, axis=None),
    param_counts_mt10[7]: scipy.stats.trim_mean(mt10['1024'], proportiontocut=0.25, axis=None),
    param_counts_mt10[8]: scipy.stats.trim_mean(mt10['2048'], proportiontocut=0.25, axis=None),
    param_counts_mt10[9]: scipy.stats.trim_mean(mt10['4096'], proportiontocut=0.25, axis=None)
}

simple_results_std = {
    param_counts_mt10[0]: np.std(mt10['SAC']),
    param_counts_mt10[1]: np.std(mt10['SM Params']),
    param_counts_mt10[3]: np.std(mt10['MOORE Params']),
    param_counts_mt10[5]: np.std(mt10['PaCo Params']),
    param_counts_mt10[7]: np.std(mt10['1024']),
    param_counts_mt10[8]: np.std(mt10['2048']),
    param_counts_mt10[9]: np.std(mt10['4096'])
}

mt50_mtrl = {
    'SM': (param_counts_mt50[1], scipy.stats.trim_mean(mt50['SM'], proportiontocut=0.25, axis=None), np.std(mt50['SM'])),
    'MOORE': (param_counts_mt50[3], scipy.stats.trim_mean(mt50['MOORE'], proportiontocut=0.25, axis=None), np.std(mt50['MOORE'])),
    'PaCo': (param_counts_mt50[5], scipy.stats.trim_mean(mt50['PaCo'], proportiontocut=0.25, axis=None), np.std(mt50['PaCo']))
}

simple_results_mt50 = {
    param_counts_mt50[0]: scipy.stats.trim_mean(mt50['SAC'], proportiontocut=0.25, axis=None),
    param_counts_mt50[1]: scipy.stats.trim_mean(mt50['SM Params'], proportiontocut=0.25, axis=None),
    param_counts_mt50[3]: scipy.stats.trim_mean(mt50['MOORE Params'], proportiontocut=0.25, axis=None),
    param_counts_mt50[5]: scipy.stats.trim_mean(mt50['PaCo Params'], proportiontocut=0.25, axis=None),
    #param_counts_mt50[7]: scipy.stats.trim_mean(mt50['1024'], proportiontocut=0.25, axis=None),
    param_counts_mt50[7]: scipy.stats.trim_mean(mt50['2048'], proportiontocut=0.25, axis=None),
    param_counts_mt50[8]: scipy.stats.trim_mean(mt50['4096'], proportiontocut=0.25, axis=None)
}

simple_results_std_mt50 = {
    param_counts_mt50[0]: np.std(mt50['SAC']),
    param_counts_mt50[1]: np.std(mt50['SM Params']),
    param_counts_mt50[3]: np.std(mt50['MOORE Params']),
    param_counts_mt50[5]: np.std(mt50['PaCo Params']),
    #param_counts_mt50[7]: np.std(mt50['1024']),
    param_counts_mt50[7]: np.std(mt50['2048']),
    param_counts_mt50[8]: np.std(mt50['4096'])
}

param_counts_mt10 = np.unique(param_counts_mt10)
params_mt10_dict = {val:idx for idx, val in enumerate(param_counts_mt10)}
print(params_mt10_dict)

param_counts_mt50 = np.unique(param_counts_mt50)
params_mt50_dict = {val:idx for idx, val in enumerate(param_counts_mt50)}

fig = plot_learning_curves(simple_results, mtrl_results, params_mt10_dict, simple_results_mt50, mt50_mtrl, params_mt50_dict)
plt.show()

exit(0)


fig = plot_results(mt10, mt50, param_counts_mt10, param_counts_mt50, mtrl_archs)
plt.show()
