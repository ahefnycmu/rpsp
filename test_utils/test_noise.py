import sys
from collections import defaultdict
from matplotlib import colors as mcolors

def test_noise():
    # plot weights from logs
    fdir = sys.argv[1]
    fnames = sys.argv[2:]
    symb1 = '--s'
    symb2 = '-o'
    models = defaultdict(lambda: ('lite-cont.pkl', mcolors.cnames['dodgerblue'], symb2),
                         {'ar_p0.1': ('arVR/VRpg/p_obs_fail/0.1_/arVR.pkl', mcolors.cnames['blue'], symb1),
                          'ar_p0.3': ('arVR/VRpg/p_obs_fail/0.3_/arVR.pkl', mcolors.cnames['green'], symb1),
                          'ar_p0.5': ('arVR/VRpg/p_obs_fail/0.5_/arVR.pkl', mcolors.cnames['brown'], symb1),
                          'arT5': ('arVR/VRpg/T_obs_fail/5_/arVR.pkl', mcolors.cnames['darkblue'], symb1),
                          'arT2': ('arVR/VRpg/T_obs_fail/2_/arVR.pkl', mcolors.cnames['black'], symb1),
                          'arT8': ('arVR/VRpg/T_obs_fail/8_/arVR.pkl', mcolors.cnames['lightblue'], symb1),
                          'arT3': ('arVR/VRpg/T_obs_fail/3_/arVR.pkl', mcolors.cnames['darkgreen'], symb1),
                          'arT4': ('arVR/VRpg/T_obs_fail/4_/arVR.pkl', mcolors.cnames['orange'], symb1),
                          'arT6': ('arVR/VRpg/T_obs_fail/6_/arVR.pkl', mcolors.cnames['lightgreen'], symb1),
                          'psrT2': ('lite-cont/VRpg/T_obs_fail/2_/lite-cont.pkl', mcolors.cnames['darkblue'], symb2),
                          'psrT5': ('lite-cont/VRpg/T_obs_fail/5_/lite-cont.pkl', mcolors.cnames['black'], symb2),
                          'psrT8': ('lite-cont/VRpg/T_obs_fail/8_/lite-cont.pkl', mcolors.cnames['lightblue'], symb2),
                          'psrT3': ('lite-cont/VRpg/T_obs_fail/3_/lite-cont.pkl', mcolors.cnames['darkgreen'], symb2),
                          'psrT4': ('lite-cont/VRpg/T_obs_fail/4_/lite-cont.pkl', mcolors.cnames['orange'], symb2),
                          'psrT6': ('lite-cont/VRpg/T_obs_fail/6_/lite-cont.pkl', mcolors.cnames['lightgreen'], symb2),
                          'aro0.1': ('arVR/VRpg/obsnoise/0.1/arVR.pkl', mcolors.cnames['black'], symb1),
                          'aro0.05': ('arVR/VRpg/obsnoise/0.05/arVR.pkl', mcolors.cnames['darkblue'], symb1),
                          'aro0.2': ('arVR/VRpg/obsnoise/0.2/arVR.pkl', mcolors.cnames['lightblue'], symb1),
                          'aro0.3': ('arVR/VRpg/obsnoise/0.3/arVR.pkl', mcolors.cnames['lightgreen'], symb1),
                          'aro0.4': ('arVR/VRpg/obsnoise/0.4/arVR.pkl', mcolors.cnames['yellow'], symb1),
                          'psro0.1': ('lite-cont/VRpg/obsnoise/0.1/lite-cont.pkl', mcolors.cnames['black'], symb2),
                          'psro0.05': ('lite-cont/VRpg/obsnoise/0.05/lite-cont.pkl', mcolors.cnames['darkblue'], symb2),
                          'psro0.2': ('lite-cont/VRpg/obsnoise/0.2/lite-cont.pkl', mcolors.cnames['lightblue'], symb2),
                          'psro0.3': ('lite-cont/VRpg/obsnoise/0.3/lite-cont.pkl', mcolors.cnames['lightgreen'], symb2),
                          'psro0.4': ('lite-cont/VRpg/obsnoise/0.4/lite-cont.pkl', mcolors.cnames['yellow'], symb2),
                          'psr_p0.2': (
                          'lite-cont/jointVROp/p_obs_fail/0.2_/lite-cont.pkl', mcolors.cnames['black'], symb2),
                          'psr_p0.1': (
                          'lite-cont/jointVROp/p_obs_fail/0.1_/lite-cont.pkl', mcolors.cnames['blue'], symb2),
                          'psr_p0.3': (
                          'lite-cont/jointVROp/p_obs_fail/0.3_/lite-cont.pkl', mcolors.cnames['green'], symb2),
                          'psr_p0.5': (
                          'lite-cont/jointVROp/p_obs_fail/0.5_/lite-cont.pkl', mcolors.cnames['brown'], symb2),
                          '1RPSP0': ('current_results/RKW/T2/Hopper-v1/lite-cont/jointVROp/T_obs_fail/',
                                     mcolors.cnames['dodgerblue'], symb2),
                          '1RPSP2': ('current_results/RKW/T2/rpsp2/Hopper-v1/lite-cont/jointVROp/T_obs_fail/',
                                     mcolors.cnames['darkblue'], symb2),
                          '1FM2+RPSP2': ('current_results/RKW/T2/fromfm/Hopper-v1/lite-cont/jointVROp/T_obs_fail/',
                                         mcolors.cnames['brown'], symb2),
                          'T2FM2': ('ar/norm/Hopper-v1/arVR/VRpg/len/500_/arVR.pkl', mcolors.cnames['red'], symb2),
                          'T2RPSP': (
                          'rpsp_var/Hopper-v1/lite-cont/jointVROp/len/500_/lite-cont.pkl', mcolors.cnames['dodgerblue'],
                          symb2),
                          'T2RPSP2': (
                          'rpsp2_var/Hopper-v1/lite-cont/jointVROp/len/500_/lite-cont.pkl', mcolors.cnames['cyan'],
                          symb2),
                          'T2FM2+RPSP2': (
                          'rpsp2fm2_var/Hopper-v1/lite-cont/jointVROp/len/500_/lite-cont.pkl', mcolors.cnames['black'],
                          symb2),
                          'T5FM2': ('Hopper-v1/arVR/VRpg/len/1000_/arVR.pkl', mcolors.cnames['red'], symb2),
                          'T5RPSP0': (
                          'rpsp/Hopper-v1/lite-cont/jointVROp/refine/0_/lite-cont.pkl', mcolors.cnames['dodgerblue'],
                          symb2),
                          'T5RPSP1000': (
                          'rpsp/Hopper-v1/lite-cont/jointVROp/refine/1000_/lite-cont.pkl', mcolors.cnames['cyan'],
                          symb2),
                          'T5RPSPref1000': (
                          'opt/Hopper-v1/lite-cont/jointVROp/rstep/0.1_/lite-cont.pkl', mcolors.cnames['black'], symb2),
                          'init1000': ('addfm_init1000/Hopper-v1/lite-cont/jointVROp/initN/1000_/lite-cont.pkl',
                                       mcolors.cnames['black'], symb2),
                          'init100': (
                          'addfm_init1000/Hopper-v1/lite-cont/jointVROp/len/1000_/lite-cont.pkl', mcolors.cnames['red'],
                          symb2),
                          'init100_interp0.1': (
                          'addfm_interp/Hopper-v1/lite-cont/jointVROp/psr_smooth/interp_0.1_/lite-cont.pkl',
                          mcolors.cnames['blue'], symb2),
                          })
    color_dict = lambda key: models[key][1]
    model_paths = lambda key: models[key][0]
    model_dirs = lambda key: '/'.join(models[key][0].split('/')[:-1]) + '/'
    shape_dict = lambda key: models[key][2]
    STEP = 30
    BEST = 10
    MAX_TRIALS = 10
    envs = ['Swimmer', 'Hopper', 'Walker2d', 'CartPole']
    lens = [300, 500, 500, 300]
    MAX_LEN = lambda e: dict([(k, v) for k, v in zip(envs, lens)])[e]
    env = 'Hopper'

    T_labels = ['arT8', 'arT6', 'arT5', 'arT4', 'arT3', 'arT2', 'psrT8', 'psrT6', 'psrT5', 'psrT4', 'psrT3',
                'psrT2']  # ]
    o_labels = ['aro0.3', 'aro0.2', 'aro0.1', 'aro0.05', 'psro0.3', 'psro0.2', 'psro0.1', 'psro0.05']
    p_labels = ['ar_p0.1', 'ar_p0.3', 'ar_p0.5', 'ar_p0.2', 'psr_p0.2', 'psr_p0.1', 'psr_p0.3', 'psr_p0.5']
    T2_labels = ['T2FM2', 'T2RPSP', 'T2RPSP2', 'T2FM2+RPSP2']
    T5_labels = ['T5FM2', 'T5RPSP0', 'T5RPSP1000', 'T5RPSPref1000']

    if False:
        R_iter, R_cum = plot_rwds(T_labels, cdict=color_dict, fdir=fdir + env + '-v1/', \
                                  model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP,
                                  MAX_TRIALS=MAX_TRIALS,
                                  title='T fail', shape_dict=shape_dict)
        plot_table(R_cum, ['FM2', 'RPSP'], ['w=8', 'w=6', 'w=5', 'w=4', 'w=3', 'w=2'], 'Signal loss (window w)', \
                   'AUC (cumulative Return $10^3$)', fdir + env + '-v1/', incr=1000, ncol=3)

    if True:
        R_iter, R_cum = plot_rwds(o_labels, cdict=color_dict, fdir=fdir + env + '-v1/', \
                                  model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP,
                                  MAX_TRIALS=MAX_TRIALS,
                                  title='p fail', shape_dict=shape_dict)
        # print o_labels
        plot_table(R_cum, ['FM2', 'RPSP'], ['0.3', '0.2', '0.1', '0.05'], 'Obstacle noise',
                   'AUC (cumulative Return $10^3$)', fdir + env + '-v1/', incr=1000)

    if False:
        for name in T2_labels:
            print fdir + model_dirs(name)
            plot_from_logs(fdir + model_dirs(name))
        plot_rwds(T2_labels, cdict=color_dict, fdir=fdir, \
                  model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                  title='T2 fail', shape_dict=shape_dict)
    if False:
        plot_rwds(T5_labels, cdict=color_dict, fdir=fdir, \
                  model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                  title='T5 fail', shape_dict=shape_dict)

    if False:
        labels = ['init1000', 'init100', 'init100_interp0.1']
        for name in labels:
            print fdir + model_dirs(name)
            plot_from_logs(fdir + model_dirs(name))
        plot_rwds(labels, cdict=color_dict, fdir=fdir, \
                  model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                  title='init_interp', shape_dict=shape_dict)

    if False:
        labels_all = []
        model_paths = lambda key: key + '/' + models[key][0]
        for name in fnames:
            print (fdir + '/' + name + '/')
            plot_from_logs(fdir + '/' + name + '/')
            labels_all.append(name)
        # print labels_all
        plot_rwds(labels_all, cdict={}, fdir=fdir, \
                  model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                  title='results', shape_dict=shape_dict)