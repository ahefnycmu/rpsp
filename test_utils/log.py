#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny, zmarinho
"""
import numpy as np

from test_utils.plot import call_plot, save_model


class Log(object):
    def __init__(self, args, filename, n=3, pred_model=None):
        self._pp = call_plot(name=filename, n=n, trial=args.trial)
        self._pred_model = pred_model
        self._args = args
        self._irate = args.irate
        self._last_err = np.inf
        self.avg_traj = []
        self._results = {'act': [], 'rewards': [], 'rwd': [], 'obs': [],
                         'mse': [], 'exp': filename, 'rng': [], 'env_states': []}

    def logger(self, i, trajs, res, track_delta=False):
        # Output stats
        C = [np.sum(t.rewards) for t in trajs]
        m = np.mean(C)
        s = np.std(C)
        wdecay = 1.0 if self._args.wdecay is None else self._args.wdecay
        wpred = self._args.grad_step if self._args.wpred is None else self._args.wpred
        rwd_coeff = self._args.wrwd if self._args.wrwd > 0.0 else self._args.wrwd_only
        wrwd = self._args.trpo_step if rwd_coeff is None else rwd_coeff
        res['best_vel_avg'] = np.mean(trajs[trajs[-1].bib].vel)
        res['best_vel_min'] = np.min(trajs[trajs[-1].bib].vel)
        res['best_vel_max'] = np.max(trajs[trajs[-1].bib].vel)
        res['best_rwd'] = np.sum(trajs[trajs[-1].bib].rewards)
        if self._pred_model is None:
            if (i % self._irate == 0):
                # self._pp.plot_single(m,s)
                self._pp.plot_traj(trajs[0], trajs[0].obs)
                self._pp.plot(np.mean(res.get('cost1_avg', 0.)), np.std(res.get('cost1_avg', 0.)),
                              np.mean(res.get('fvel_avg', 0.0)), np.std(res.get('fvel_avg', 0.0)),
                              m, s, False, label_2='vel')
        else:
            normalizer = float(wpred * wdecay ** i) if wpred > 0.0 else wdecay ** i
            emse = (res.get('total_cost', 0.) - wrwd * res.get('reinf_cost_avg', 0.)) / normalizer
            R = [emse]
            self._results['mse'].append(R)
            if track_delta:
                ##track difference between avg trajectory for exploration evaluation
                avg = np.zeros((self._args.numtrajs, self._args.len, trajs[0].obs.shape[1]))
                for k, t in enumerate(trajs):
                    avg[k, :t.obs.shape[0], :] = t.obs / float(t.obs.shape[0])
                self.avg_traj.append(np.sum(avg, axis=0))
                print('\t\tdelta_batch_avg:{} delta_prev_avg:{}'.format(
                    np.linalg.norm(np.mean([(t.obs - self.avg_traj[-1][:t.obs.shape[0]]) ** 2], axis=0)),
                    np.linalg.norm(np.mean([(t.obs - 0.0
                                             if len(self.avg_traj) < 2
                                             else self.avg_traj[-2][:t.obs.shape[0]]) ** 2], axis=0)),
                    ))
            self._last_err = np.mean(np.copy(R))
            if (i % self._irate == 0):
                try:
                    reinf = res.get('trpo_cost', 0.)
                except KeyError:
                    reinf = res.get('cost2_avg', 0.)
                self._pp.plot(np.mean(res.get('cost1_avg', 0.)), np.std(res.get('cost1_avg', 0.)),
                              np.mean(res.get('fvel_avg', 0.0)), np.std(res.get('fvel_avg', 0.0)),
                              m, s, False, label_2='vel')
                tpred = self._pred_model.traj_predict_1s(trajs[0].states, trajs[0].act)
                self._pp.plot_traj(trajs[0], tpred)
        print 'reg:{} psr_step:{} rwd_w:{} past:{} fut:{}'.format(self._args.reg, self._args.grad_step, self._args.wrwd,
                                                                  self._args.past, self._args.fut)
        print '\t\t\t\t\t\t' + '\t\t\t\t\t\t'.join(['{}={}\n'.format(k, res.get(k, 0.0)) for k in res.keys()])
        self._results['rewards'].append([np.sum(t.rewards) for t in trajs])

        if (i % 50 == 0):
            self._results['env_states'].append([trajs[trajs[-1].bib].env_states])
            self._results['rwd'].append([trajs[trajs[-1].bib].rewards])
            self._results['act'].append([trajs[trajs[-1].bib].act])
            self._results['rng'].append([trajs[trajs[-1].bib].rng])
            self._results['obs'].append([trajs[trajs[-1].bib].obs])
        if (i % (self._args.prate) == 0):
            # save pickle results
            save_model(self._args.method + '_trial%d' % self._args.trial, self._args.flname, self._results, self._args)
        return False