#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:38:44 2017

@author: zmarinho, ahefny
"""
import matplotlib

from test_policy_network import test_policy_continuous
from test_utils.plot import *

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.rcParams.update({'xtick.labelsize': 15})
matplotlib.rcParams.update({'ytick.labelsize': 15})


class structtype():
    pass


''' Run the given model N times saves results in pickle inside filename '''
def run_Nmodel(args, filename, N=5, plot=True, loadfile=''):
    try:
        print ('Loading existing model...')
        load_file = (filename, loadfile)[loadfile != '']
        model_results, nargs = load_model(args.method + '.pkl', load_file)
        N = len(model_results)

        for i in xrange(N):
            print 'trial %d out of %d' % (i, N)
            args.trial = i
            if args.monitor is not None:
                args.monitor += '%d' % args.trial
            setattr(args, 'params', model_results[i]['params'])
            test_policy_continuous(args, filename)
        return
    except IOError:
        setattr(args, 'loadfile', '')
        print 'Model non-existent. Running experiment:'
        model_results = []
        for i in xrange(N):
            print 'trial %d' % i, args.iter
            args.trial = i
            results = test_policy_continuous(args, filename)
            if results is not None:
                model_results.append(results)  # rewards per iter and mse
            save_model(args.method, filename, model_results, args)
            if plot and len(model_results) > 0:
                plot_trials(model_results, filename)

    return model_results, args



if __name__ == '__main__':
    import sys

    ''' 
    print plots for different models
    usage:
    file_dir modelpath1/pickle modelpath2/pickle ...
    in order displayed by labels could be less .
    python stats_test.py final_results/CartPole/ full_observable+reactive/obsVR3trials_s10.pkl partial_observable+reactive/obsVR3trials_s10.pkl AR5_partial+reactive/arVR3trials_s10.pkl 
    plots :
    file_dir: used to save plots 
    STEP: resolution of each unit (iterations)
    BEST: plot best k trials
    MAX_LEN: maximum length of episode
    '''

    STEP = 10
    BEST = 10
    MAX_TRIALS = 10
    # timesteps=[1000,500,1000,200]
    lens = [500, 500, 500, 200]
    fdir = sys.argv[1]
    dirfile = fdir[:]
    symb1 = '-s'
    symb2 = '-o'
    symb3 = '-^'
    symb4 = '-p'

    envs = ['Hopper', 'Swimmer', 'Walker2d', 'CartPole']
    envs_labels = ['FM', 'GRU', 'RPSP-Alt',
                   'RPSP-VRPG']  # 'FM',    #,'VRPG+obs','Alt+obs'] #, 'RPSP']#, 'RPSPicg'] #'AR1',
    minimal_labels = ['GRU_16', 'GRU_re', 'Alt']
    # envs_labels = [ 'RPSP', 'RPSP_l2','RPSPicg']#,'RPSP_NS', ]
    # opt_labels = ['RPSP-VRPG', 'RPSP-Alt', 'TRPO']
    RPSP_labels = ['VRPG+obs', 'Alt+obs', 'VRPG', 'Alt']  # ,'TRPO']
    init_labels = ['RPSP_rand', 'RPSP_PSR']
    AR_labels = ['FM%d' % ard for ard in [1, 2, 5]]  # +['FM1','full AR1']
    cg_labels = ['adam', 'adagrad', 'RMSProp', 'sgd']
    gru_labels = ['GRU_16', 'GRU_32', 'GRU_64', 'GRU_128']
    # RPSPgru_labels = {'Hopper':['VRPG+obs', 'Alt', 'VRPG-gru+obs', 'Alt-gru+obs'],
    #                  'Swimmer':['VRPG+obs', 'VRPG-gru+obs', 'Alt+obs','Alt-gru+obs'],\
    #                  'CartPole':['VRPG', 'VRPG-gru+obs',  'Alt', 'Alt-gru+obs'],\
    #                  'Walker2d':['VRPG+obs','VRPG-gru+obs', 'Alt+obs', 'Alt-gru+obs']}#,'TRPO']
    best_models = {'Hopper': {'random_PSR': ('random_PSR', mcolors.cnames['black'], symb4),
                              'fix_PSR': ('Alt_fix_PSR', mcolors.cnames['brown'], symb2),
                              'reactive_PSR': ('Alt_reactive_loss', mcolors.cnames['grey'], symb3),
                              'VRPG': ('VRPG+obs', mcolors.cnames['dodgerblue'], symb1),
                              'Alt': ('Alt', mcolors.cnames['limegreen'], symb1),
                              'RPSP': ('Alt', mcolors.cnames['dodgerblue'], symb1),
                              'VRPG-GRU': ('VRPG-gru+obs', mcolors.cnames['darkblue'], symb3),
                              'Alt-GRU': ('Alt-gru+obs', mcolors.cnames['darkgreen'], symb3),
                              'reg_GRU': ('Alt-gru+obs', mcolors.cnames['orange'], symb3),
                              'GRU': ('GRU', mcolors.cnames['red'], symb3),
                              'reactive_GRU': ('react_GRU', mcolors.cnames['purple'], symb3),
                              'PSD': ('PSD', mcolors.cnames['brown'], symb3)},
                   'Swimmer': {'random_PSR': ('random_PSR', mcolors.cnames['black'], symb4),
                               'fix_PSR': ('VRPG_fix_PSR', mcolors.cnames['brown'], symb2),
                               'reactive_PSR': ('VRPG_reactive_loss', mcolors.cnames['grey'], symb3),
                               'VRPG': ('VRPG+obs', mcolors.cnames['dodgerblue'], symb1),
                               'Alt': ('Alt+obs', mcolors.cnames['limegreen'], symb1),
                               'RPSP': ('VRPG+obs', mcolors.cnames['dodgerblue'], symb1),
                               'VRPG-GRU': ('VRPG-gru+obs', mcolors.cnames['darkblue'], symb3),
                               'Alt-GRU': ('Alt-gru+obs', mcolors.cnames['darkgreen'], symb3),
                               'reg_GRU': ('Alt-gru+obs', mcolors.cnames['orange'], symb3),
                               'GRU': ('GRU', mcolors.cnames['red'], symb3),
                               'reactive_GRU': ('react_GRU', mcolors.cnames['purple'], symb3),
                               'PSD': ('PSD', mcolors.cnames['brown'], symb3)},
                   'CartPole': {'random_PSR': ('random_PSR', mcolors.cnames['black'], symb4),
                                'fix_PSR': ('Alt_fix_PSR', mcolors.cnames['brown'], symb2),
                                'reactive_PSR': ('Alt_reactive_loss', mcolors.cnames['grey'], symb3),
                                'VRPG': ('VRPG', mcolors.cnames['dodgerblue'], symb1),
                                'Alt': ('Alt', mcolors.cnames['limegreen'], symb1),
                                'RPSP': ('VRPG', mcolors.cnames['dodgerblue'], symb1),
                                'VRPG-GRU': ('VRPG-gru+obs', mcolors.cnames['darkblue'], symb3),
                                'Alt-GRU': ('Alt-gru+obs', mcolors.cnames['darkgreen'], symb3),
                                'reg_GRU': ('Alt-gru+obs', mcolors.cnames['orange'], symb3),
                                'GRU': ('GRU', mcolors.cnames['red'], symb3),
                                'reactive_GRU': ('react_GRU', mcolors.cnames['purple'], symb3),
                                'PSD': ('PSD', mcolors.cnames['brown'], symb3)},
                   'Walker2d': {'random_PSR': ('random_PSR', mcolors.cnames['black'], symb4),
                                'fix_PSR': ('Alt_fix_PSR', mcolors.cnames['brown'], symb2),
                                'reactive_PSR': ('Alt_reactive_loss', mcolors.cnames['grey'], symb3),
                                'VRPG': ('VRPG+obs', mcolors.cnames['dodgerblue'], symb1),
                                'Alt': ('Alt+obs', mcolors.cnames['limegreen'], symb1),
                                'RPSP': ('Alt+obs', mcolors.cnames['dodgerblue'], symb1),
                                'VRPG-GRU': ('VRPG-gru+obs', mcolors.cnames['darkblue'], symb3),
                                'Alt-GRU': ('Alt-gru+obs', mcolors.cnames['darkgreen'], symb3),
                                'reg_GRU': ('Alt-gru+obs', mcolors.cnames['orange'], symb3),
                                'GRU': ('GRU', mcolors.cnames['red'], symb3),
                                'reactive_GRU': ('react_GRU', mcolors.cnames['purple'], symb3),
                                'PSD': ('PSD', mcolors.cnames['brown'], symb3)},
                   }
    filter_labels = ['GRU', 'reg_GRU', 'RPSP']  # ,'reactive_GRU','PSD']
    VRPGgru_labels = ['VRPG+obs', 'VRPG', 'VRPG-gru+obs', 'VRPG-gru']  # ,'TRPO']
    RPSPgru_labels = ['VRPG', 'Alt', 'VRPG-GRU', 'Alt-GRU']
    Altgru_labels = ['Alt+obs', 'Alt', 'Alt-gru+obs', 'Alt-gru']  # ,'TRPO']
    reactive_labels = ['Alt_reactive_loss', 'VRPG_reactive_loss', 'Alt_fix_PSR', 'VRPG_fix_PSR', 'Alt', 'VRPG']
    reg_labels = ['RPSP', 'fix_PSR', 'reactive_PSR', 'random_PSR']
    noise_labels = ['FM2 ', 'Alt ']  # ,'VRPG ']

    rmax = 100
    from matplotlib import colors as mcolors

    models = {'full AR1': ('full_norm/obsVR.pkl', mcolors.cnames['blue']),
              'FM1': ('AR1/arVR.pkl', mcolors.cnames['black']),
              'FM2': ('AR2/arVR.pkl', mcolors.cnames['lawngreen']),
              'FM3': ('AR3/arVR.pkl', mcolors.cnames['green']),
              'FM4': ('AR4/arVR.pkl', mcolors.cnames['blue']),
              'FM5': ('AR5/arVR.pkl', mcolors.cnames['olive']),
              'FM10': ('AR10/arVR.pkl', mcolors.cnames['orange']),
              'AR1b5': ('partial_obs+AR/obsVR.pkl', mcolors.cnames['yellowgreen']),
              # 'FM5':('ar_norm/arVR.pkl', mcolors.cnames['olive']),
              'FM': ('AR/arVR.pkl', mcolors.cnames['orange']),
              # 'FM1':('partial_norm/obsVR.pkl',  mcolors.cnames['red']),
              # 'FM1':('partial_obs+reactive/obsVR.pkl', mcolors.cnames['yellow']),
              # 'FM5':('AR+reactive/arVR.pkl', mcolors.cnames['olive']),
              # 'FM5':('AR5/arVR.pkl', mcolors.cnames['olive']),
              'lstm_old': ('lstm_old/lstmVR.pkl', mcolors.cnames['tomato']),
              'GRU': ('lr1e-2/best/results_{}.csv'.format(BEST), mcolors.cnames['tomato']),
              # 'GRU':('GRU/gru.pkl'.format(BEST),mcolors.cnames['tomato']),
              'VR': ('PSRnet_VRPG/lite-cont.pkl', mcolors.cnames['black']),
              'RPSP-VRPG': ('VRPG_BEST/lite-cont.pkl', mcolors.cnames['dodgerblue']),
              'VRPG': ('VRPG/lite-cont.pkl', mcolors.cnames['dodgerblue'], symb3),
              'RPSPicg': ('PSRnet_icg/lite-cont.pkl', mcolors.cnames['darkblue']),
              'RPSP_PSR': ('PSRnet_joint/lite-cont.pkl', mcolors.cnames['dodgerblue']),
              'RPSP-Alt': ('ALT_BEST/lite-cont.pkl', mcolors.cnames['darkgreen']),
              'Alt': ('Alt/lite-cont.pkl', mcolors.cnames['darkgreen'], symb3),
              'RPSP': ('VRPG/lite-cont.pkl', mcolors.cnames['darkgreen']),
              'VRPG+obs': ('VRPG+obs/lite-cont.pkl', mcolors.cnames['turquoise']),
              'Alt+obs': ('Alt+obs/lite-cont.pkl', mcolors.cnames['limegreen']),
              '+rwd': ('+rwd/lite-cont.pkl', mcolors.cnames['chocolate']),
              '+rwd+obs': ('+rwd+obs/lite-cont.pkl', mcolors.cnames['magenta']),
              'RPSP_rand': ('PSRnet_AltOp_rand/lite-cont.pkl', mcolors.cnames['darkblue']),
              'TRPO': ('PSRnet_TRPO/lite-cont.pkl', mcolors.cnames['cyan']),
              'RPSP_NS': ('kbr_MIA/lite-cont.pkl', mcolors.cnames['darkblue']),
              'RPSP_l2': ('kbr_state_norm/lite-cont.pkl', mcolors.cnames['lightblue']),
              'adam': ('adam/lite-cont.pkl', mcolors.cnames['gold']),
              'RMSProp': ('RMSProp/lite-cont.pkl', mcolors.cnames['chocolate']),
              'adagrad': ('adagrad/lite-cont.pkl', mcolors.cnames['magenta']),
              'adadelta': ('adadelta/lite-cont.pkl', mcolors.cnames['blue']),
              'sgd': ('sgd/lite-cont.pkl', mcolors.cnames['green']),
              'GRU_re': ('minimal/results_{}.csv'.format(BEST), mcolors.cnames['orange']),
              'GRU_16': ('lr1e-2/gru16/results_{}.csv'.format(BEST), mcolors.cnames['brown']),
              'GRU_32': ('lr1e-2/gru32/results_{}.csv'.format(BEST), mcolors.cnames['blue']),
              'GRU_64': ('lr1e-2/gru64/results_{}.csv'.format(BEST), mcolors.cnames['red']),
              'GRU_128': ('lr1e-2/gru128/results_{}.csv'.format(BEST), mcolors.cnames['orange']),
              'VRPG-gru': ('VRPG-gru/gru.pkl', mcolors.cnames['purple']),
              'Alt-gru': ('Alt-gru/gru.pkl', mcolors.cnames['darkgreen']),
              'VRPG-gru+obs': ('VRPG-gru+obs/gru.pkl', mcolors.cnames['turquoise']),
              'VRPG-nopred': ('VRPG_nopred/lite-cont.pkl', mcolors.cnames['dodgerblue']),
              'Alt-nopred': ('Alt_nopred/lite-cont.pkl', mcolors.cnames['darkblue']),
              'Alt-gru+obs': ('Alt-gru+obs/gru.pkl', mcolors.cnames['limegreen']),
              'Alt_reactive_loss': ('fix_psr_no/Alt/lite-cont.pkl', mcolors.cnames['green'], symb2),
              'VRPG_reactive_loss': ('fix_psr_no/VRPG/lite-cont.pkl', mcolors.cnames['darkblue'], symb2),
              'Alt_fix_PSR': ('fix_psr/Alt/lite-cont.pkl', mcolors.cnames['limegreen'], symb1),
              'VRPG_fix_PSR': ('fix_psr/VRPG/lite-cont.pkl', mcolors.cnames['lightblue'], symb1),
              'FM2 ': ('/obsnoise/AR/{}/arVR.pkl', mcolors.cnames['orange'], symb2),
              'Alt ': ('/obsnoise/Alt/{}/lite-cont.pkl', mcolors.cnames['limegreen'], symb2),
              'VRPG ': ('/obsnoise/VRPG/{}/lite-cont.pkl', mcolors.cnames['dodgerblue'], symb2),
              'random_PSR': ('random_PSR/lite-cont.pkl', mcolors.cnames['black'], symb3),
              'PSD': ('PSD/psd.csv', mcolors.cnames['black'], symb3),
              'react_GRU': ('VRPG-gru_nopred/gru.pkl', mcolors.cnames['black'], symb3),

              }
    color_dict = lambda key: models[key][1]
    noise_paths = lambda noise: lambda key: models[key][0].format(noise)
    model_paths = lambda key: models[key][0]
    MAX_LEN = lambda e: dict([(k, v) for k, v in zip(envs, lens)])[e]
    shape_dict = lambda key: '-o'
    shape_dict2 = lambda key: '-^' if 'gru' in key else '-o'
    shape_dict3 = lambda key: models[key][2]
    best_colors = lambda env: lambda key: best_models[env][key][1]
    best_shape = lambda env: lambda key: best_models[env][key][2]
    best_paths = lambda env: lambda key: models[best_models[env][key][0]][0]
    # envs = ['Swimmer']
    # i=1;
    # '''plot for GRUs minimal comparison'''
    # plot_rwds( minimal_labels, cdict=color_dict, fdir=fdir+envs[i]+'-v1/', \
    #      model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
    #      title=envs[i]+r' minimal', step = max_lims[envs[i]][1])
    # exit()

    # noise_levels=[0.3,0.2,0.1,0.05]
    noise_levels = [0.1, 0.2, 0.3]
    max_lims = {'Hopper': (1000, 1000), 'Swimmer': (130, 500), 'CartPole': (200, 200), 'Walker2d': (1000, 1000)}  # 1400
    ''' Experiments for each environment'''

    R_iter = [];
    for i in xrange(len(envs)):
        AUC = [];
        fdir = dirfile
        env = envs[i]
        fdir_add = 'R0/new/' if envs[i] == 'Hopper' else ''
        print '--------------------HERE', envs[i], fdir_add
        # best_paths = lambda key: models[best_models[env][key][0]][0]
        if False and env == 'Hopper':
            vel_new = get_log_key(fdir + envs[i] + '-v1/VRPG/', string='fvel_avg', plot=False)
            vel_old = get_log_key(fdir + envs[i] + '-v1/VRPG_old/', string='fvel_avg', plot=False)
            print len(vel_new), len(vel_old)

            fig = plot_trials(vel_new, fdir + envs[i] + '-v1/', f=lambda it: vel_new[it], name='Hopper_opt', shape='-s',
                              step=20)
            plt.ylabel(envs[i] + ' Cumulative reward $R_{iter}$')
            fig = plot_trials(vel_old, fdir + envs[i] + '-v1/', f=lambda it: vel_old[it], name='Hopper_Gym', fig=fig,
                              shape='-o', step=20)

        if False:
            plot_rwds(reactive_labels, cdict=color_dict, fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                      model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                      title=envs[i] + 'Prediction as regularization $R_{iter}$ ', shape_dict=shape_dict3, ncol=3,
                      kfold=max_lims[envs[i]][1] * MAX_TRIALS)

        if True:
            R_cumulative = []
            f, ax = plt.subplots(nrows=1, ncols=len(noise_levels), figsize=(15, 6), sharey=True)
            ax[0].set_ylabel(envs[i] + ' $R_{iter}$ ')
            for j, noise in enumerate(noise_levels):
                R_iter, R_cum, auc = plot_rwds(noise_labels, cdict=color_dict, fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                                               model_paths=noise_paths(noise), BEST=BEST, MAX_LEN=MAX_LEN(envs[i]),
                                               STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                                               shape_dict=shape_dict3, ylim=[-10, 800],  # max_lims[envs[i]][0]],\
                                               f=f, ax=ax[j], legend=(j == 1), kfold=max_lims[envs[i]][1] * MAX_TRIALS)
                # title=envs[i]+' Observation noise ',
                R_cumulative.extend(R_cum)
                plt.subplots_adjust(wspace=0.1, top=0.9, bottom=0.2, left=0.08, right=0.99, )
                ax[j].set_title("noise $\sigma$=" + str(noise), position=(0.5, 0.92), fontsize=20)
            f.savefig(fdir + envs[i] + '-v1/' + envs[i] + 'noiseobs_rwdperiter_best%d_step%d.png' % (BEST, STEP),
                      dpi=300)
            # print 'donenoise'
            # plot_table(R_cumulative, noise_labels, noise_levels, envs[i]+' Obstacle noise AUC', 'AUC (cumulative Return $10^3$)', fdir+env+'-v1/', incr=1000)

        if False:
            '''Experiments for RPSP/gru minimal comparison'''
            plot_rwds(RPSPgru_labels, cdict=best_colors(envs[i]), fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                      model_paths=best_paths(envs[i]), BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP,
                      MAX_TRIALS=MAX_TRIALS,
                      title=envs[i] + ' GRU Filter comparison $R_{iter}$ ', step=max_lims[envs[i]][1],
                      shape_dict=best_shape(envs[i]),
                      kfold=max_lims[envs[i]][1] * MAX_TRIALS)

        if True:
            '''Experiments for RPSP/gru minimal comparison'''
            x, x, aucurve = plot_rwds(filter_labels, cdict=best_colors(envs[i]),
                                      fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                                      model_paths=best_paths(envs[i]), BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP,
                                      MAX_TRIALS=MAX_TRIALS,
                                      title=envs[i] + ' PSR vs. GRU filtering $R_{iter}$ ', step=max_lims[envs[i]][1],
                                      shape_dict=best_shape(envs[i]), ylim=[-10, max_lims[envs[i]][0]],
                                      kfold=max_lims[envs[i]][1] * MAX_TRIALS, ncol=5)

            Rgru = np.asarray(aucurve, dtype=float)

        if True:
            x, x, aucurve = plot_rwds(reg_labels, cdict=best_colors(envs[i]), fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                                      model_paths=best_paths(envs[i]), BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP,
                                      MAX_TRIALS=MAX_TRIALS,
                                      title=envs[i] + ' Filter regularization $R_{iter}$ ', step=max_lims[envs[i]][1],
                                      shape_dict=best_shape(envs[i]), ylim=[-10, max_lims[envs[i]][0]],
                                      kfold=max_lims[envs[i]][1] * MAX_TRIALS)
            Rfilter = np.asarray(aucurve, dtype=float)

        if True:

            print env
            bestmodels, R_cum, aucurve = plot_rwds(envs_labels, cdict=color_dict,
                                                   fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                                                   model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]),
                                                   STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                                                   title=envs[i] + ' $R_{iter}$ ', step=max_lims[envs[i]][1],
                                                   shape_dict=shape_dict,
                                                   kfold=max_lims[envs[i]][1] * MAX_TRIALS)
            print 'done print'
            R_iter.append(bestmodels)
            R = np.asarray(aucurve, dtype=float)

            AUC.extend(zip(np.mean(R, axis=1), np.std(R, axis=1) / np.sqrt(R.shape[1])))
            AUC.extend(zip(np.mean(Rgru, axis=1), np.std(Rgru, axis=1) / np.sqrt(Rgru.shape[1])))
            AUC.extend(zip(np.mean(Rfilter, axis=1), np.std(Rfilter, axis=1) / np.sqrt(Rfilter.shape[1])))
            import scipy.stats

            print 'done. Computing AUC'


            def print_ttest(labels, R, m, j):
                print 't-test (', labels[m], labels[-j], ')=',
                t, prob = scipy.stats.ttest_ind(R[m, :], R[-j, :], axis=0, equal_var=False)
                print '[', t, ', ', prob, '] '
                return


            for j in [1, 2]:
                for m in xrange(len(envs_labels) - 1):
                    print_ttest(envs_labels, R, m, j)
            print '\n'
            print_ttest(filter_labels, Rgru, 1, 0)
            for k in [0, 1, 3]:
                print_ttest(reg_labels, Rfilter, 2, k)
                # Swimmer
                # t-test ( FM RPSP-VRPG )= [ -0.676586869332 ,  0.507653821772 ]
                # t-test ( GRU RPSP-VRPG )= [ -11.4782215848 ,  9.9484555701e-07 ]
                # t-test ( RPSP-Alt RPSP-VRPG )= [ -3.10454361915 ,  0.00933872173406 ]
                # t-test ( FM RPSP-Alt )= [ 1.73508240067 ,  0.111134237192 ]
                # t-test ( GRU RPSP-Alt )= [ -20.2985459977 ,  2.4045809148e-09 ]
                # Hopper
                # t-test ( FM RPSP-VRPG )= [ -1.97496384693 ,  0.0638921448438 ]
                # t-test ( GRU RPSP-VRPG )= [ -2.98367426585 ,  0.0121047687853 ]
                # t-test ( RPSP-Alt RPSP-VRPG )= [ 2.25405603636 ,  0.0382107798284 ]
                # t-test ( FM RPSP-Alt )= [ -3.92498523554 ,  0.00123896806466 ]
                # t-test ( GRU RPSP-Alt )= [ -4.909161505 ,  0.000577310309778 ]
                # Walker2d
                # t-test ( FM RPSP-VRPG )= [ -0.0793340586349 ,  0.937667000948 ]
                # t-test ( GRU RPSP-VRPG )= [ -2.9657440937 ,  0.00870691843556 ]
                # t-test ( RPSP-Alt RPSP-VRPG )= [ 2.23688724131 ,  0.0412540309747 ]
                # t-test ( FM RPSP-Alt )= [ -2.05494189388 ,  0.0601025184967 ]
                # t-test ( GRU RPSP-Alt )= [ -6.51606587482 ,  5.3677973935e-06 ]
                # CartPole
                # t-test ( FM RPSP-VRPG )= [ -4.49013758712 ,  0.00103201257316 ]
                # t-test ( GRU RPSP-VRPG )= [ 1.23597901022 ,  0.232353991956 ]
                # t-test ( RPSP-Alt RPSP-VRPG )= [ -0.358340067592 ,  0.726097854222 ]
                # t-test ( FM RPSP-Alt )= [ -8.06880999537 ,  6.44596048562e-07 ]
                # t-test ( GRU RPSP-Alt )= [ 1.95090137725 ,  0.0740240842398 ]
            print np.round(np.asarray(AUC)[[0, 1, 2, 3, 5, 9], :] / 1000.0, decimals=1)
            print reg_labels, filter_labels

        if False:
            '''plot for GRUs'''
            plot_rwds(gru_labels, cdict=color_dict, fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                      model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                      title=envs[i] + r' GRU $R_{iter}$ ', step=max_lims[envs[i]][1], shape_dict=shape_dict,
                      kfold=max_lims[envs[i]][1] * MAX_TRIALS)
        if True:
            '''Experiments with varying window'''
            plot_rwds(AR_labels, cdict=color_dict, fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                      model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                      title=envs[i] + ' FM $R_{iter}$ ', step=max_lims[envs[i]][1], shape_dict=shape_dict,
                      kfold=max_lims[envs[i]][1] * MAX_TRIALS)
            '''Experiments for RPSP variants'''
            plot_rwds(RPSP_labels, cdict=color_dict, fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                      model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                      title=envs[i] + ' RPSP variants $R_{iter}$ ', step=max_lims[envs[i]][1], shape_dict=shape_dict,
                      kfold=max_lims[envs[i]][1] * MAX_TRIALS)

            '''Experiments for RPSP/gru minimal comparison'''
            plot_rwds(Altgru_labels, cdict=color_dict, fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                      model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                      title=envs[i] + ' AltvsGRU comparison', step=max_lims[envs[i]][1], shape_dict=shape_dict2)
            '''Experiments for RPSP/gru minimal comparison'''
            plot_rwds(VRPGgru_labels, cdict=color_dict, fdir=fdir + envs[i] + '-v1/' + fdir_add, \
                      model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(envs[i]), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                      title=envs[i] + ' VRPGvsGRU comparison', step=max_lims[envs[i]][1], shape_dict=shape_dict2)

    if False:
        LABELS = envs_labels
        lens = [300, 500, 500, 200]
        steps = [3, 5, 5, 2]
        f = plt.figure()
        scaled_rwds = np.zeros((len(LABELS), 100))
        stds = np.zeros((len(LABELS), 100))
        for l in xrange(len(R_iter)):
            max_rwd = np.max([np.max(R_iter[l][m][0]) for m in xrange(len(LABELS))])
            for m in xrange(len(LABELS)):
                scaled_auc_rewards = np.cumsum(R_iter[l][m][0][0:lens[l]:steps[l]]) / float(max_rwd)
                scaled_rwds[m, :] += scaled_auc_rewards
                stds[m, :] += np.asarray(R_iter[l][m][1][0:lens[l]:steps[l]]) / float(max_rwd)

        plot_models(zip(scaled_rwds, stds), names=LABELS, length=100, \
                    fdir=fdir, step=1, figname='AUC', \
                    ylabel='AUC', xlabel='experience (% timesteps)', \
                    cdict=color_dict, timestep=max_lims[envs[i]][1], \
                    shape_dict=lambda key: '-', stds=1.0, linewidth=2)

#     f=plt.figure()
#     t = np.arange(100)
#     for i in xrange(len(LABELS)):
#         plt.errorbar(t, scaled_rwds[i], yerr=stds[i]/3.0, label=LABELS[i],color=color_dict(LABELS[i]), capsize=0)
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#           frameon=False, ncol=6, columnspacing=0.2, handletextpad=0.0, fontsize=18)
#     #plt.title(title)
#     plt.xlabel('experience (% timesteps)')
#     plt.ylabel('AUC')
#     #plt.ylim([0,plt.ylim()[1]])
#     f.savefig(fdir+'AUC.png')


if False:
    '''Experiments with random initialization'''
    env = envs[0]
    plot_rwds(init_labels, cdict=color_dict, fdir=fdir + env + '-v1/', \
              model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
              title='RPSP Initialization')

    '''Experiments for RPSP opts'''
    env = envs[0]
    plot_rwds(opt_labels, cdict=color_dict, fdir=fdir + env + '-v1/', \
              model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
              title='RPSP optimizers')
    '''Experiments with varying window'''
    env = envs[0]
    plot_rwds(AR_labels, cdict=color_dict, fdir=fdir + env + '-v1/', \
              model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
              title='AR window')

    '''Experiments with varying Gopt'''
    env = envs[0]
    plot_rwds(cg_labels, cdict=color_dict, fdir=fdir + env + '-v1/', \
              model_paths=model_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
              title='Gradient Optimizers')

    ''' Experiments with obsnoise'''
    fnoise_dir = '../new_final_results/Swimmer-v1/NOISE/'
    noise_paths = {}
    noise_vals = np.arange(0.0, 1.0, 0.1)
    [noise_paths.update({'joint_%.1f' % noise: '%.1f/lite-cont/lite-cont.pkl' % noise, \
                         'obs_%.1f' % noise: '%.1f/obsVR/obsVR.pkl' % noise, \
                         'AR_%.1f' % noise: '%.1f/arVR/arVR.pkl' % noise}) for noise in noise_vals]
    colors_noise1 = plt.cm.winter(np.linspace(0, 1, len(noise_vals)))
    colors_noise2 = plt.cm.autumn(np.linspace(0, 1, len(noise_vals)))
    colors_noise3 = plt.cm.cool(np.linspace(0, 1, len(noise_vals)))

    ncolor_dict = {}
    ncolor_dict.update(dict(
        [(k, colors_noise1[-1]) for (i, k) in enumerate(list(filter(lambda k: k[:4] == 'join', noise_paths.keys())))]))
    ncolor_dict.update(dict(
        [(k, colors_noise2[-1]) for (i, k) in enumerate(list(filter(lambda k: k[:3] == 'obs', noise_paths.keys())))]))
    ncolor_dict.update(dict(
        [(k, colors_noise3[-1]) for (i, k) in enumerate(list(filter(lambda k: k[:2] == 'AR', noise_paths.keys())))]))
    cn_dict = lambda key: ncolor_dict.get(key)
    n_paths = lambda key: noise_paths.get(key)

    nkeys = [['joint_%.1f' % noise, 'obs_%.1f' % noise, 'AR_%.1f' % noise] for noise in noise_vals];
    for i, i_nkeys in enumerate(nkeys):
        plot_rwds(i_nkeys, cdict=cn_dict, fdir=fnoise_dir, \
                  model_paths=n_paths, BEST=BEST, MAX_LEN=MAX_LEN(env), STEP=STEP, MAX_TRIALS=MAX_TRIALS,
                  title='Noise_performance%.1f' % noise_vals[i])
