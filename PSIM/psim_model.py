from psim import Controlled_PSIM
from models import FilteringModel
from psim import compute_action_prob 
from environments import GymEnvironment
import numpy as np
import cPickle
from NN_policies import DiscretePolicy
from NN_policies import ContinuousPolicy
from kernel_ridge_regression import RidgeRegression
from kernel_ridge_regression import RFF_RidgeRegression
from IPython import embed
from policy_learn import learn_model_policy
from NN_policies import VR_Reinforce_PolicyUpdater
from environments import PartiallyObservableEnvironment

class PSIM_FilteringModel(FilteringModel):

    def __init__(self, Learner, obs_dim, a_dim,  
                    k = 5,  win_len = 5000,  
                    PSIM_iter = 5, weighted = True, 
                    normalization = True, feature_type = 'Linear',
                    discrete = True, bootstrap = False):

        self._psim_model = Controlled_PSIM(obs_dim = obs_dim, a_dim = a_dim, k = k,
                win_len = win_len, PSIM_iter = PSIM_iter,  
                weighted = weighted, normalization = normalization,
                feature_type = feature_type, bootstrap = bootstrap);
        
        self._psim_model.set_learner(Learner = Learner);
        self.belief = np.copy(self.psim_model.belief);
        self.policy = None;
        self.discrete = discrete; #if discrete: a_dim equals to the number of actions.
        
        self.a_dim = a_dim;
        self.obs_dim = obs_dim;

    @property
    def psim_model(self):
        return self._psim_model;

    @property
    def state_dimension(self):
        return self.psim_model.belief_dimension; #+ self.obs_dim;

    @property
    def input_dimension(self):
        return self.psim_model.input_dimension;

    def set_policy(self, policy):
        self.policy = policy;


    def get_params(self):
        return self.psim_model.learner.get_params();
    
    def set_params(self,A):
        self.psim_model.learner.set_params(A);


    def update(self, trajs, iter_id = 0):
        if self.discrete is True:
            [trajs_obs, trajs_act, trajs_prob]=compute_action_prob(self.policy, trajs, 
                            num_actions = self.a_dim);
        else:
            [trajs_obs, trajs_act, trajs_prob]=compute_action_prob(self.policy, trajs);
        
        #if iter_id == 0:
        self.psim_model.add_new_trajs(trajs_obs = trajs_obs, 
                    trajs_act = trajs_act,
                    trajs_prob = trajs_prob);
        self.psim_model.InferenceMachine(iter_id);
        
    def reset(self, first_observation):
        self.belief = np.copy(self.psim_model.belief); #return the initial belief stored at psim.
        return self.belief;
        #return np.concatenate([self.belief, first_observation],axis=0); 
    
    def update_state(self, o, a): #filtering one step:
        if self.discrete:
            control = np.zeros(self.a_dim);
            control[int(a)] = 1.;
        else:
            control = a;
        input_to_learner = self.psim_model._form_input(self.belief, 
                    curr_act = control, 
                    curr_obs = o);

        if self.psim_model.learner.initialized() is True:
            self.belief = self.psim_model.learner.predict(input_to_learner.reshape(1,-1))[0];
        else:
            self.belief = np.zeros(self.belief.shape[0]);
        return self.belief;
        #return np.concatenate([self.belief, o],axis=0);


if __name__ == '__main__':

    seeds = 200;
    np.random.seed(seeds);
    env = GymEnvironment('CartPole-v0');
    #env = PartiallyObservableEnvironment(GymEnvironment('CartPole-v0'), np.array([0,2]))
    #env = PartiallyObservableEnvironment(GymEnvironment('Hopper-v1',discrete = False), 
    #            np.array([0,1,2,3]));
    discrete = False;
    #env._base.env.seed(seeds)
    #env = PartiallyObservableEnvironment(GymEnvironment('Acrobot-v1'), np.array([0,1,2,3]));
    env.reset();
    (obs_dim, a_dim) = env.dimensions;
    output_dim = env.action_info[0];
    
    #embed()

    #Learner = RidgeRegression(Ridge = 1e-6);
    Learner = RFF_RidgeRegression(Ridge = 1e-6, bwscale = 1.);
    psim_model = PSIM_FilteringModel(Learner = Learner, obs_dim = obs_dim, 
                a_dim = output_dim, k = 2,
                win_len = 100, PSIM_iter = 5, weighted = False, 
                normalization = False, 
                feature_type = 'Linear', discrete = discrete, bootstrap = True);

    #embed()

    #pi = DiscretePolicy(x_dim = psim_model.state_dimension, output_dim = output_dim,
    #    num_layers = 1, nh = [16]);
    pi = ContinuousPolicy(x_dim = psim_model.state_dimension,output_dim = output_dim,
            num_layers = 1, nh = [64]);

    psim_model.set_policy(policy = pi);

    PiUpdator = VR_Reinforce_PolicyUpdater(policy = pi, lr = 1e-3);
    rr = learn_model_policy(policy_updater = PiUpdator, 
            model = psim_model, environment = env,
            max_traj_len = 100, num_trajs = 100, num_iter = 100);


        
    
    

