#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:16:14 2017

@author: ahefny
"""

import numpy as np
import theano
import theano.tensor as T
import theano.typed_list

from policy.NN_policies import RidgeRegression
from rpsp.policy_opt.SGD_opt import optimizers
from rpsp.policy_opt.policy_learn import BasePolicyUpdater


class VR_Reinforce_RNN_PolicyUpdater(BasePolicyUpdater):
    def __init__(self, *args, **kwargs):
        self.max_traj_len = args[1];
        self.num_trajs = args[2];
        self._policy = args[0];
        self._psr = self._policy._psrnet
        self.lr = kwargs.pop('lr', 1e-2);
        self._beta_reinf = kwargs.pop('beta_reinf', 0.0)
        self._beta_pred = kwargs.pop('beta_pred', 1.0)
        self._beta_decay = kwargs.pop('beta_pred_decay', 1.0)
        self._beta_only_reinf = kwargs.pop('beta_only_reinf', 1.0)
        self._optimizer = kwargs.pop('cg_opt', 'adam')
        self.gamma = kwargs.pop('gamma', 0.99);
        self.gamma_seq = np.array([self.gamma ** (i) for i in xrange(10000)]);
        self.discount = kwargs.pop('discount', 1.0)
        self.baseline = kwargs.pop('baseline', None);
        self._t_beta_pred = theano.shared(self._beta_pred)
        if self.baseline is not None:
            self.linear_reg = RidgeRegression(Ridge=1e-7);
            self.obs_set = None;
            self.ctgs_set = None;
            self.max_num = 50000;

        symbolic_list_trajs_X = [];  # symbolic representation of a list of trajectories (obs)
        symbolic_list_trajs_XF = [];  # symbolic representation of a list of trajectories (obs features)
        symbolic_list_trajs_UF = [];  # simbolic representation of a list of control features (U)
        symbolic_list_trajs_U = [];  # simbolic representation of a list of controls (U)
        symbolic_list_trajs_ctg = [];  # simbolic representation of a list of trajectory ctgs.
        for i in xrange(0, self.num_trajs):
            symbolic_list_trajs_X.append(T.matrix('trajx_{}'.format(i)));
            symbolic_list_trajs_XF.append(T.matrix('trajxf_{}'.format(i)));
            symbolic_list_trajs_UF.append(T.matrix('trajuf_{}'.format(i)));
            if self._policy.discrete == True:
                symbolic_list_trajs_U.append(T.imatrix('traju_{}'.format(i)));
            else:
                symbolic_list_trajs_U.append(T.matrix('traju_{}'.format(i)));
            symbolic_list_trajs_ctg.append(T.vector('ctg_{}'.format(i)));

        self._t_list_traj_X = T.stack(symbolic_list_trajs_X);  # 3d tensor
        self._t_list_traj_XF = T.stack(symbolic_list_trajs_XF);  # 3d tensor
        self._t_list_traj_UF = T.stack(symbolic_list_trajs_UF);  # 3d tensor
        self._t_list_traj_U = T.stack(symbolic_list_trajs_U);  # 2d tensor if discrete, 3d otherwise
        self._t_list_ctg = T.stack(symbolic_list_trajs_ctg);  # 2d tensor.

        self._t_traj_lengths = T.ivector('traj_lengths')

        self.build_graph()
        self.cost_trajs = theano.function(
            inputs=[self._t_list_traj_XF, self._t_list_traj_X,
                    self._t_list_traj_UF, self._t_list_traj_U,
                    self._t_list_ctg, self._t_traj_lengths],
            outputs=self._t_avg_ctg,
            allow_input_downcast=True,
            on_unused_input='warn');

        self.cost = theano.function(inputs=[self._t_list_traj_XF, self._t_list_traj_X,
                                            self._t_list_traj_UF, self._t_list_traj_U,
                                            self._t_list_ctg, self._t_traj_lengths],
                                    outputs=[self._t_cost, self._t_reinf_loss],
                                    allow_input_downcast=True,
                                    on_unused_input='warn')

    def build_graph(self):

        self._t_lr = T.scalar('lr');
        self._t_cost = T.scalar('cost')
        self._t_reinf_loss = T.scalar('err')
        self._t_avg_ctg, self._t_reinf_loss = self._t_cost_fn(self._t_list_traj_XF, self._t_list_traj_X,
                                                              self._t_list_traj_UF, self._t_list_traj_U,
                                                              self._t_list_ctg, self._t_traj_lengths);

        self._t_policy_update = optimizers[self._optimizer](T.mean(self._t_reinf_loss),
                                                            self._policy._policy.params,
                                                            self._t_lr * self._beta_only_reinf);
        self._t_psr_update = optimizers[self._optimizer](self._t_avg_ctg,
                                                         self._policy._psrnet.params, self._t_lr) + [
                                 (self._t_beta_pred, self._t_beta_pred * self._beta_decay)];
        if len(self._policy._psrnet._params_proj) > 0:
            print 'adding pca projection parameters'
        self.gradient_descent = theano.function(
            inputs=[self._t_list_traj_XF, self._t_list_traj_X,
                    self._t_list_traj_UF, self._t_list_traj_U,
                    self._t_list_ctg, self._t_traj_lengths, self._t_lr],
            outputs=[self._t_avg_ctg, self._t_reinf_loss],
            updates=self._t_policy_update + self._t_psr_update,
            allow_input_downcast=True,
            on_unused_input='warn');
        self._t_cost, self._t_reinf_loss = self._t_single_traj_cost(self._t_list_traj_XF[0], self._t_list_traj_X[0],
                                                                    self._t_list_traj_UF[0], self._t_list_traj_U[0],
                                                                    self._t_list_ctg[0], self._t_traj_lengths[0])

    @property
    def policy(self):
        return self._policy

    def set_psr(self, psrnet):
        self.policy.reset_psrnet(psrnet)
        self._psr = self.policy._psrnet
        self.build_graph()
        return

    def _t_single_traj_cost(self, XF, X, UF, U, ctg, length):
        H = self._policy._t_compute_states(XF, UF)
        valid_len = length

        error = self._psr.tf_1smse_wprestate(H[0:valid_len - 1], UF[1:valid_len], X[1:valid_len])
        probs = self._policy._t_compute_prob(H[0:valid_len - 1], U[1:valid_len])
        probs = T.clip(probs, 1e-12, np.inf)

        reinf = T.mean(T.log(probs) * ctg[1:valid_len])
        cost = self._beta_reinf * reinf + self._t_beta_pred * error
        return cost, reinf  # the average loss this particular trajectory

    def _t_cost_fn(self, XFs, Xs, UFs, Us, ctgs, lengths):
        # Xs: a list of 2d matrix;
        # Us: a list of 1d vector or 2d matrix;
        # ctgs: a list of 1d vector;

        ccs, _ = theano.scan(fn=self._t_single_traj_cost,
                             sequences=[XFs, Xs, UFs, Us, ctgs, lengths], n_steps=XFs.shape[0],
                             name='_t_cost_fn')

        return T.mean(ccs[0]), ccs[1];

    def _padding(self, trajs, num_trajs=None):  # trajs: obs, state, u, reward.#U is in one-hot encoding.

        tensor_traj_X = np.zeros((num_trajs,
                                  self.max_traj_len, trajs[0][1].shape[1]));

        tensor_traj_XF = np.zeros((num_trajs, self.max_traj_len, trajs[0].obs_feat.shape[1]))
        tensor_traj_UF = np.zeros((num_trajs, self.max_traj_len, trajs[0].act_feat.shape[1]))
        traj_lengths = np.array([t.length for t in trajs], dtype=np.int)

        if self._policy.discrete is True:
            tensor_traj_U = np.ones((num_trajs, self.max_traj_len, self._policy.output_dim)) * (-np.inf);
        else:
            tensor_traj_U = np.ones((num_trajs, self.max_traj_len,
                                     trajs[0][2].shape[1])) * (-np.inf);
        tensor_traj_ctg = np.zeros((num_trajs, self.max_traj_len));
        trajs_tc = [];

        obs = trajs[0].obs[:-1];
        for i in xrange(0, num_trajs):
            traj_length = trajs[i].length
            tensor_traj_X[i, :traj_length, :] = trajs[i].obs
            tensor_traj_XF[i, :traj_length, :] = trajs[i].obs_feat
            tensor_traj_UF[i, :traj_length, :] = trajs[i].act_feat

            if self._policy.discrete is True:
                tensor_traj_U[i, 0:traj_length, :] = np.zeros((traj_length, self._policy.output_dim))
                tensor_traj_U[i, np.arange(traj_length), trajs[i].act[:, 0].astype(int)] = 1.
            else:
                tensor_traj_U[i, 0:traj_length] = trajs[i].act

            tmp_ctgs = np.array([-np.sum(trajs[i].rewards[j:] * self.gamma_seq[0:len(trajs[i].rewards[j:])]) for j in
                                 xrange(traj_length)])
            ctgs = np.copy(tmp_ctgs)

            trajs_tc.append(-np.sum(trajs[i].rewards))  # convert reward to cost.
            if self.baseline is not None:
                if self.baseline == 'obs':
                    obs = tensor_traj_X[i, :traj_length, :]
                elif self.baseline == 'AR':
                    d = tensor_traj_X.shape[2]
                    w = self._psr._rffpsr._past
                    obs = np.zeros((traj_length, d * w), dtype=float)
                    for j in xrange(w):
                        obs[:traj_length - i, d * j:d * (j + 1)] = tensor_traj_X[i, i:traj_length, :]
                self.obs_set = obs if self.obs_set is None else np.concatenate((self.obs_set, obs), axis=0);
                self.ctgs_set = ctgs if self.ctgs_set is None else np.concatenate((self.ctgs_set, tmp_ctgs));
                assert self.obs_set.shape[0] == self.ctgs_set.shape[0], 'shape mismatch padding'
                if self.obs_set.shape[0] > self.max_num:
                    self.obs_set = self.obs_set[-self.max_num:];
                    self.ctgs_set = self.ctgs_set[-self.max_num:];

                self.linear_reg.fit(input_X=self.obs_set, Y=self.ctgs_set.reshape(-1, 1));
                pred_ctgs = self.linear_reg.predict(input_X=obs);
                ctgs = ctgs - pred_ctgs.flatten();
            tensor_traj_ctg[i, :traj_length] = ctgs

        return [tensor_traj_X, tensor_traj_XF, tensor_traj_UF, tensor_traj_U, tensor_traj_ctg, trajs_tc, traj_lengths];

    def update(self, trajs):
        assert not np.isnan(self._policy.get_params()).any(), 'rnn model policy paramters nan'

        for t in trajs:
            t.obs_feat = self._policy._fext_obs.process(t.obs)
            t.act_feat = self._policy._fext_act.process(t.act)
        assert self.num_trajs == len(trajs), 'wrong number of trajectories length'
        [list_traj_X, list_traj_XF, list_traj_UF, list_traj_U, list_traj_ctg, list_traj_cc,
         traj_lengths] = self._padding(trajs, num_trajs=self.num_trajs);
        output, grad_err = self.gradient_descent(list_traj_XF, list_traj_X, list_traj_UF,
                                                 list_traj_U, list_traj_ctg, traj_lengths, self.lr);
        assert not np.isinf(np.mean(grad_err)), 'reinforce loss too large'
        return {'model_cost_avg': np.mean(list_traj_cc),
                'model_cost_std': np.std(list_traj_cc),
                'reinf_cost_avg': np.mean(grad_err),
                'reinf_cost_std': np.std(grad_err),
                'total_cost': output};

    def get_states(self, trajs):
        assert not np.isnan(self._policy.get_params()).any(), 'rnn model policy paramters nan'
        for t in trajs:
            t.obs_feat = self._policy._fext_obs.process(t.obs)
            t.act_feat = self._policy._fext_act.process(t.act)
        [list_traj_X, list_traj_XF, list_traj_UF, list_traj_U, list_traj_ctg, list_traj_cc,
         traj_lengths] = self._padding(trajs, num_trajs=len(trajs));

        list_states = []
        for i, t in enumerate(trajs):
            states = self._psr._get_pre_states(list_traj_XF[i, :traj_lengths[i], :],
                                               list_traj_UF[i, :traj_lengths[i], :])
            list_states.append(states)
        return np.vstack(list_states)
