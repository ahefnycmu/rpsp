import numpy as np
import NN_policies
from environments import GymEnvironment, PartiallyObservableEnvironment
import psr_lite.rffpsr
import psr_lite.feat_extractor
import psr_lite.rffpsr_rnn
import psr_lite.psrlite_policy
from models import ObservableModel
import policies
import psr_lite.psr_lite_vrpg
from policy_learn import learn_policy

np.random.seed(200);
env = GymEnvironment('CartPole-v0');
#env = partial_obs_Gym_CartPole_Env('CartPole-v0');
#env = GymEnvironment('Acrobot-v1');
#env = GymEnvironment('Swimmer-v1', discrete = False);
#env = PartiallyObservableEnvironment(GymEnvironment('Swimmer-v1', discrete = False), np.array([0,1,2,3]))
#env = PartiallyObservableEnvironment(GymEnvironment('CartPole-v1'),np.array([0,2]));
env.reset();
(x_dim, a_dim) = env.dimensions;
print x_dim, a_dim;
output_dim = env.action_info[0];
print output_dim
#this model could be replace by PSR or PSIM later.
model = ObservableModel(obs_dim = x_dim); 

max_traj_length = 100;
num_trajs = 50;

pi_exp = policies.RandomDiscretePolicy(output_dim)

exp_trajs = env.run(model, pi_exp, 10, 50)
col_trajs = [(t.obs.T, t.act.T) for t in exp_trajs]
X_obs = [c[0] for c in col_trajs]
X_act = [c[1] for c in col_trajs]
  
feats = psr_lite.feat_extractor.create_RFFPCA_featureset(1000,20)
feats['act'] = psr_lite.feat_extractor.IndicatorFeatureExtractor()
psr = psr_lite.rffpsr.RFFPSR(2, 2, 50, feature_set=feats)
psr.train(X_obs, X_act)
psrrnn = psr_lite.rffpsr_rnn.RFFPSR_RNN(psr)

pi_react = NN_policies.DiscretePolicy(x_dim = psr.state_dim, output_dim = output_dim, num_layers = 1, nh = [5]);

pi = psr_lite.psrlite_policy.RFFPSRNetworkPolicy(psrrnn, pi_react, np.array([0]))

#embed()

#pi = RNN_Discrete_Policy(x_dim = x_dim, a_dim = a_dim, output_dim=output_dim, nh = 16, LSTM = True);
#pi = RNN_Continuous_Policy(x_dim=x_dim, a_dim = a_dim, output_dim=output_dim,nh = 64, LSTM = True);
#pi = ContinuousPolicy(x_dim = x_dim, output_dim = output_dim, num_layers = 1, nh = [64]);
print 'build updater ... ',
#PiUpdator = NN_policies.VR_Reinforce_PolicyUpdater(policy = pi, lr = 1e-2);
PiUpdator = psr_lite.psr_lite_vrpg.VR_Reinforce_RNN_PolicyUpdater(policy = pi, 
                        max_traj_length = max_traj_length,
                        num_trajs = num_trajs, lr = 1e-4, beta_reinf=1.0, beta_pred=10.0, beta_pred_decay=0.5);
print 'done'

learn_policy(PiUpdator, model, env,
            max_traj_len = max_traj_length, num_trajs = num_trajs, 
            num_iter = 100);