# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:38:44 2017

@author: zmarinho, ahefny
"""
from rpsp.run.test_policy_network import run_policy_continuous
from rpsp.run.test_utils.plot import load_model, save_model, plot_trials


class structtype():
    pass


def run_Nmodel(args, filename, N=5, plot=True, loadfile=''):
    """
    Run the given model N times saves results in pickle inside filename
    @param args: command line arguments
    @param filename: filename to store results
    @param N: run model N times
    @param plot: plot results if true
    @param loadfile: load pretrianed model
    @return: model results, arguments
    """
    try:
        print ('Loading existing model...')
        load_file = (filename, loadfile)[loadfile != '']
        model_results, nargs = load_model(args.method + '.pkl', load_file)
        N = len(model_results)
        for i in xrange(N):
            print('trial %d out of %d' % (i, N))
            args.trial = i
            if args.monitor is not None:
                args.monitor += '%d' % args.trial
            setattr(args, 'params', model_results[i]['params'])
            run_policy_continuous(args, filename)
        return

    except IOError:
        setattr(args, 'loadfile', '')
        print ('Model non-existent. Running experiment:')
        model_results = []

        for i in xrange(N):
            print ('trial %d' % i, args.iter)
            args.trial = i
            results = run_policy_continuous(args, filename)
            if results is not None:
                model_results.append(results)  # rewards per iter and mse
            save_model(args.method, filename, model_results, args)
            if plot and len(model_results) > 0:
                plot_trials(model_results, filename)

    return model_results, args

