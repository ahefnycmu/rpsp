import numpy as np
from IPython import embed
import copy
import cPickle 
from NN_policies import DiscretePolicy
from kernel_ridge_regression import *
from sklearn.linear_model import Ridge


def compute_action_prob(policy, trajs, num_actions = None): 
    #trajs: generated from environment.run. 
    #trajs[i] = (obs[:j,:],states[:j,:],act[:j,:],rwd[:j,:])
    #policy: a stochastic policy that can compute the probablity of an action given a corresponding state. 
    N_traj = len(trajs);
    trajs_obs = [];
    trajs_state = [];
    trajs_prob = [];
    trajs_act = [];
    for i in xrange(N_traj):
        #traj_obs = trajs[i][0][:,[0,2]];
        traj_obs = trajs[i][0];
        traj_state = trajs[i][1];
        if num_actions is None:
            traj_act = trajs[i][2];
        else:
            N = trajs[i][2].shape[0];
            traj_act = np.zeros((N, num_actions));
            traj_act[np.arange(N), trajs[i][2].reshape(N).astype(int)] = 1.;
            
        traj_prob = np.zeros(traj_obs.shape[0]);
        traj_prob[0] = 1.0/num_actions if num_actions is not None else 1.0;
        for step in xrange(1, traj_obs.shape[0]):
            if policy.discrete is True:
                traj_prob[step] = policy.compute_action_prob(traj_state[step-1], np.argmax(traj_act[step]));      
            else:
                traj_prob[step] = policy.compute_action_prob(traj_state[step-1], (traj_act[step]));      
        trajs_obs.append(traj_obs);
        trajs_state.append(traj_state);
        trajs_prob.append(traj_prob);
        trajs_act.append(traj_act); #vector: one-hot encoding if discrete action. 
    return [trajs_obs, trajs_act, trajs_prob];

class online_normalization(object):
    def __init__(self, x_dim):
        self.mean = np.zeros(x_dim);
        self.stds = np.ones(x_dim); 
        self.batch_num = 0.0;

    def normalize(self, Input, update = True):
        if update is True:
            input_m = np.mean(Input, axis=0);
            if self.batch_num == 0.0:
                self.mean = input_m;
                self.stds = (np.mean(Input**2,axis=0) -self.mean**2)**0.5;
            else:
                self.mean = (self.mean * self.batch_num + input_m) / (self.batch_num + 1.);
                inputs_std = (np.mean(Input**2,axis=0) - self.mean**2)**0.5;
                self.stds = (self.stds * self.batch_num + inputs_std) / (self.batch_num + 1.);
            self.batch_num = self.batch_num + 1;
        self.stds[np.abs(self.stds) < 1e-8] = 1.
        return (Input - self.mean)*(1./self.stds);
        

class Controlled_PSIM(object):
    '''
    This class implements controlled PSIM, using importance sampling techniques. 
    '''
    def __init__(self, obs_dim, a_dim, k = 5, win_len = 5000,  PSIM_iter = 10, weighted = True, 
                    normalization = True, feature_type = 'Linear', bootstrap = False):
        self.k = k;
        self.PSIM_iter = PSIM_iter;
        self.trajs_obs = []; #window
        self.trajs_act = [];
        self.trajs_prob = []; #window.
        self.window_len = win_len;
        self.weighted = weighted;
        self.normalize = normalization;
        self.normalizer = None;

        self.best_learner = None;
        self.best_normalizer = None;
        self.min_pred_error = np.inf;
        self.learner = None;

        self.bootstrap = bootstrap;
        
        if feature_type == 'Linear' or 'Quadratic':
            self.feature_type = feature_type;
        else:
            print 'current does not support feature type {}'.format(feat_ure_type);
            assert False;
        
        self.obs_dim = obs_dim;
        self.a_dim = a_dim;
        tmp_belief = self._feature(np.random.rand(self.k, self.obs_dim),
                        np.random.rand(self.k, self.a_dim));
        self._belief = np.zeros(tmp_belief.shape[0]); #initlize the belief. 
        self._tmp_input = self._form_input(tmp_belief, np.zeros(self.a_dim), np.zeros(self.obs_dim));

    @property
    def belief_dimension(self):
        return self._belief.shape[0];

    @property
    def belief(self):
        return self._belief;
    
    @property
    def input_dimension(self):
        return self._tmp_input.shape[0];

    def initialize_learner(self):
        self.learner.initialize(x_dim = self.input_dimension, y_dim = self.belief_dimension);

    def add_new_trajs(self, trajs_obs, trajs_act, trajs_prob):
        win_len = np.min([self.window_len, len(self.trajs_obs) + len(trajs_obs)]);
        self.trajs_obs = (self.trajs_obs + trajs_obs)[-win_len:];
        self.trajs_act = (self.trajs_act + trajs_act)[-win_len:];
        self.trajs_prob = (self.trajs_prob + trajs_prob)[-win_len:];

        self.obs_dim = trajs_obs[0].shape[1];
        self.num_acts = trajs_act[0].shape[1];

    def set_learner(self, Learner):
        self.learner = Learner;
        self.initialize_learner();

    def _feature(self, seq_obs, seq_act):
        #expected: seq_obs and seq act are a 2d matrix. 
        fea = np.hstack((seq_obs[0:self.k].ravel(), seq_act[0:self.k].ravel(), 1.));
        if self.feature_type == 'Quadratic':
            fea_2 = fea**2;
            return np.hstack((fea,fea_2));
        elif self.feature_type == 'Linear':
            return fea;
        
    def _weighted_feature(self, seq_obs, seq_act, seq_prob, lam = 1e-8):
        prob = np.prod(seq_prob[0:self.k]) + lam;
        unweighted_fea = self._feature(seq_obs, seq_act);
        return unweighted_fea / prob;

    def initialize_belief(self, seqs_obs, seqs_act, seqs_prob):
        N = len(seqs_obs);
        for i in xrange(N):
            if self.weighted is True:
                tmp = self._weighted_feature(seqs_obs[i], seqs_act[i], seqs_prob[i]);
            else:
                tmp = self._feature(seqs_obs[i], seqs_act[i]);
            if i == 0:
                b0 = tmp;
            else:
                b0 = b0 + tmp;
        return b0 / N; 

    def _form_input(self, est_belief, curr_act, curr_obs):
        #expected: est_belief, curr_act and curr_obs are vector. 
        return np.hstack((est_belief, curr_act,curr_obs));

    def rolling_in(self, trajs_obs, trajs_act, trajs_prob, 
            initialization = False):
        N = len(trajs_obs);
        reg_inputs = [];
        reg_outputs = [];
        trajs_belief = [];
        est_belief = self.belief;

        pred_err = 0.0
        num = 1.0;
        for i in xrange(N):
            traj_obs = trajs_obs[i];
            traj_prob = trajs_prob[i];
            traj_act = trajs_act[i];
            traj_belief = [];

            est_belief = self.belief; #initlization the inital belief.
            for t in xrange(1, traj_obs.shape[0] - self.k + 1):
                curr_act = traj_act[t-1];
                curr_obs = traj_obs[t-1];
                curr_input = self._form_input(est_belief, curr_act, curr_obs);
                traj_belief.append(est_belief);
                
                if self.weighted:
                    next_emp_b = self._weighted_feature(traj_obs[t:], traj_act[t:],traj_prob[t:]);
                else:
                    next_emp_b = self._feature(traj_obs[t:],traj_act[t:]);
                reg_inputs.append(curr_input);
                reg_outputs.append(next_emp_b);
                
                if initialization is True:
                    est_belief = next_emp_b; #this is for DAgger's initialization. 
                else:
                    curr_input = curr_input if self.normalize is False else self.normalizer.normalize(curr_input,update = False)
                    est_belief = self.learner.predict(curr_input.reshape(1,-1))[0]; #otherwise: just predict. 
                    pred_err = pred_err + np.linalg.norm(est_belief[:self.k*self.obs_dim] - next_emp_b[:self.k*self.obs_dim]);
                    #print pred_err
                    num = num + 1;
            
            trajs_belief.append(np.array(traj_belief));
        return [reg_inputs, reg_outputs, trajs_belief, pred_err/num];

    def initlization(self, trajs_obs, trajs_act, trajs_prob):
        [inputs, outputs] = self.rolling_in(trajs_obs, trajs_act, trajs_prob,
                            initialization = True)[0:2];

        emp_b_len = np.mean(np.linalg.norm(np.array(outputs)[:,:self.k*self.obs_dim], axis = 1));
        print "the l2 norm of the empirical belief is {}".format(emp_b_len);
        if self.normalize is True:
            self.normalizer = online_normalization(np.array(inputs).shape[1]);
            inputs = self.normalizer.normalize(np.array(inputs));
        self.learner.fit(np.array(inputs), np.array(outputs));

    def InferenceMachine(self, iter_id):
        print len(self.trajs_obs)
        if self.bootstrap is True:
            perm = np.random.choice(len(self.trajs_obs),len(self.trajs_obs), replace = True);
        else:
            perm = np.random.choice(len(self.trajs_obs),len(self.trajs_obs), replace = False);

        
        tmp_traj_obs = [self.trajs_obs[i] for i in perm];
        tmp_traj_act = [self.trajs_act[i] for i in perm];
        tmp_traj_prob= [self.trajs_prob[i] for i in perm];

        self._belief = self.initialize_belief(tmp_traj_obs, tmp_traj_act, tmp_traj_prob);
        self.initlization(tmp_traj_obs, tmp_traj_act, tmp_traj_prob);
        Agg_inputs = [];
        Agg_outputs = [];
        
        self.min_pred_error = np.inf;
        for i in xrange(self.PSIM_iter):
            [inputs,outputs, beliefs, pred_err] = self.rolling_in(
                            trajs_obs = tmp_traj_obs, trajs_act = tmp_traj_act, 
                            trajs_prob = tmp_traj_prob,
                            initialization = False);
            print "at iteration {}, the L2 prediction error is {}".format(i, pred_err);
            if pred_err <= self.min_pred_error:
                self.min_pred_error = pred_err;
                self.best_learner = copy.deepcopy(self.learner);
                self.best_normalizer = copy.deepcopy(self.normalizer) if self.normalize is True else None;
                print "the best learner is updated...";

            #if self.normalize is True:
            #    inputs = self.normalizer.normalize(np.array(inputs));
            Agg_inputs = Agg_inputs + inputs;
            Agg_outputs = Agg_outputs + outputs;
            self.learner.fit(np.array(Agg_inputs), np.array(Agg_outputs));
        
        #update the learner and normalizer:
        self.learner = copy.deepcopy(self.best_learner);
        self.normalizer = copy.deepcopy(self.best_normalizer);
    
    def filtering_on_trajs(self, trajs_obs, trajs_act, trajs_prob):
        [ip, op, trajs_beliefs, p_err] = self.rolling_in(trajs_obs, trajs_act, trajs_prob);
        return trajs_beliefs;



if __name__ == '__main__':

    [trajs_1, pi_1] = cPickle.load(open('cartpole_data_iter_10.save', 'rb'));
    [trajs_2, pi_2] = cPickle.load(open('/Users/lettue/Dropbox/PSIM_action/cartpole_data_iter_20.save', 'rb'));

    rr_1 = compute_action_prob(pi_1, trajs_1);
    rr_2 = compute_action_prob(pi_2, trajs_2);

    cPSIM = Controlled_PSIM(k = 3, win_len = 5000, PSIM_iter = 10, weighted = False, 
                normalization = False, feature_type = 'Linear');
    #linear_learner = Ridge(alpha = 1e2);
    learner = RidgeRegression(Ridge = 1e-4);
    #learner = RFF_RidgeRegression(Ridge = 1e-3, bwscale = 1.);
    
    cPSIM.set_learner(learner);
    cPSIM.add_new_trajs(rr_1[0], rr_1[1], rr_1[2]);

    #test psim first (no importance weighting):
    cPSIM.InferenceMachine();

    pred_beliefs = cPSIM.filtering_on_trajs(rr_2[0], rr_2[1], rr_2[2]);




        






