import matlab
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()

def np2mat(mat):
    assert mat.ndim == 2    
    return matlab.double([[float(y) for y in x] for x in mat])

def test_lds(fut, past, train_obs, train_act, test_obs, test_act, burn_in, error_fn = None):
    train_obs = [np2mat(x) for x in train_obs]
    train_act = [np2mat(x) for x in train_act]
    test_obs = [np2mat(x) for x in test_obs]
    test_act = [np2mat(x) for x in test_act]
    
    if error_fn is None:
        error_fn = lambda x,y : np.sum((x-y) ** 2)
    
    est_obs = eng.lds(train_obs,train_act,test_obs,test_act,float(fut),float(past),{'p':'best'})
    
    M = len(est_obs)
    mse = np.zeros(fut)
    
    for h in xrange(fut):
        for i in xrange(M):
            mse_i = 0.0
            
            N = len(test_obs[i])
            for (x,y) in zip(test_obs[i][burn_in+fut:N], est_obs[i][h][burn_in+fut:N]):        
                mse_i += error_fn(np.array(x), np.array(y))
                
            mse[h] += mse_i / (N-burn_in-fut)
            
    mse /= M
            
    return est_obs, mse





