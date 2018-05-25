import keras
from modular_rl import discount
from sklearn.preprocessing import StandardScaler
from IPython import embed
import numpy as np


class Expert_Vf(object):
    """
    class that stores a pre-trained V^* estimator (keras NN) and 
    it's corresponding scaler (sklearn scaler)
    """
    def __init__(self, net = None, scaler = None):
        self.net = net;
        self.scaler = scaler;
 
    def predict(self, raw_obs, time_step):
        raw_obs_t = np.concatenate([raw_obs, np.array([time_step])],axis=0)
        trans_obs = self.scaler.transform(raw_obs_t.reshape(1,-1))
        return self.net.predict(trans_obs)[:]
 
    def predict_path(self, path):
        #use raw observations for expert value function estimator
        ob_no = self.preproc(path["raw_observation"]) 
        ob_no = self.scaler.transform(ob_no)  
        return self.net.predict(ob_no)[:] #return an 1-d array.

    def preproc(self, ob_no):
        return np.concatenate([ob_no, np.arange(len(ob_no)).reshape(-1,1)], axis=1)


def fit_value_function_k_steps(f, gamma, paths, k = 1):
    print "optimizing TD error with {}-step ahead...".format(k)
    gammas = np.array([gamma**i for i in xrange(k)])
    all_obs = [];
    y_targets = [];
    print "obs dim: {}".format(paths[0]["observation"].shape[1])
    for path in paths:
        all_obs += list(path["observation"][0:-k])
        #compute pred values:
        pref_vs = (gamma**k)*f.predict(path['observation'][k:])[:];
        #compute k-step total rewards:
        for i in xrange(len(path['observation'][0:-k])):
            rew_seq = np.array(path['reward'][i:i+k])
            t_rew = gammas.dot(rew_seq)  #r_i+\gamma r_{i+1}+...
            pref_vs[i] += t_rew  # r_i+\gamma r_{i+1} +... + \gamma^k f(x).
        y_targets+= list(pref_vs);
    
    perm = np.random.choice(len(y_targets),len(y_targets), replace = False)
    all_obs = np.array(all_obs)[perm]
    y_targets = np.array(y_targets)[perm]

    f.fit(all_obs, y_targets, batch_size=32, epochs=5, validation_split=0.1);

def fit_value_function(f, gamma, paths):
    #whiten the dataset:
    for path in paths:
        path["observation"] = np.concatenate([path["observation"], 
            np.arange(len(path["observation"])).reshape(-1,1)],axis=1)

    all_obs = np.concatenate([path["observation"] for path in paths],axis=0)
    scaler = StandardScaler()
    scaler.fit(np.array(all_obs))

    all_obs = [];
    y_targets = [];
    for path in paths:
        path["observation"] = scaler.transform(path['observation'])
        path["return"] = discount(path["reward"],gamma)
        all_obs += list(path["observation"])
        y_targets += list(path["return"])
    
    all_obs = np.array(all_obs)
    y_targets = np.array(y_targets).reshape(-1,1)
    
    perm = np.random.choice(y_targets.shape[0], y_targets.shape[0], replace=False)
    all_obs = all_obs[perm]
    y_targets = y_targets[perm]
    
    f.fit(all_obs, y_targets, batch_size = 32, epochs = 10, validation_split=0.1)

    for k in reversed(range(1,50, 5)):
        fit_value_function_k_steps(f, gamma, paths, k = k)
    
    hat_V_star = Expert_Vf(net = f, scaler = scaler)
    return hat_V_star #for future use. 


    


