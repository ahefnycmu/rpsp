from policies import BasePolicy
import numpy as np
import theano
import theano.tensor as T
from scipy import stats
from IPython import embed
import math


class RNN_Policy(BasePolicy):
    def __init__(self, x_dim, output_dim, nh, params = None, rng=None):
        self.x_dim = x_dim; 
        self.output_dim = output_dim;
        self.nh = nh;
        self.rng = rng;
        self.params = [];
        
        self.W_x=theano.shared(name='W_x',value= np.sqrt(2./(self.x_dim+self.nh))
                        *np.random.randn(self.nh,self.x_dim).astype(theano.config.floatX));
        self.W_h=theano.shared(name='W_h',value= np.sqrt(2./(self.nh+self.nh))
                        *np.random.randn(self.nh,self.nh).astype(theano.config.floatX));
        self.hb = theano.shared(name='hn',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
        self.h0 = theano.shared(name='h0',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
        
        self.W = theano.shared(name='W',value=np.sqrt(2./(self.output_dim+self.nh))
                        *np.random.randn(self.output_dim,self.nh).astype(theano.config.floatX));
        self.b = theano.shared(name='b',
                        value=np.zeros(self.output_dim).astype(theano.config.floatX));
        
        self.params = [self.W_x,self.W_h,self.hb,self.h0,self.W,self.b];

        if params is not None:
            assert len(params) == len(self.params);
            for i in xrange(len(params)):
                self.params[i].set_value(params[i]);


class RNN_Discrete_Policy(RNN_Policy):
    def __init__(self, x_dim, output_dim, nh, params = None, rng=None):
        RNN_Policy.__init__(self, x_dim, output_dim, nh, params, rng=rng);
        self.discrete = True;

        self._t_mem = T.vector('mem');
        self._t_state = T.vector('mem');

        [self._t_updated_mem, self._t_prob] = self._t_single_step_forward(self._t_state,self._t_mem);
        self._update_mem=theano.function(inputs = [self._t_state,self._t_mem],
                                    outputs=self._t_updated_mem, allow_input_downcast=True);
        self._compute_prob=theano.function(inputs = [self._t_state,self._t_mem],
                                    outputs=self._t_prob, allow_input_downcast=True);
        self.reset();

    def reset(self):
        self.mem = self.h0.get_value();
        return self.mem;

    def update_mem(self, x):
        self.mem = self._update_mem(x, self.mem);  

    def _t_single_step_forward(self, x, h):
        h_tp1 = T.nnet.relu(T.dot(h,self.W_h.T)+T.dot(x,self.W_x.T)+self.hb, alpha = 0.01);
        y = T.nnet.softmax(T.dot(h_tp1,self.W.T) + self.b)[0];
        return [h_tp1, y];
        
    def _t_compute_prob(self, X, U = None): 
        [hs,ys],_ = theano.scan(fn=self._t_single_step_forward,
                                sequences=X, outputs_info=[self.h0,None],
                                n_steps = X.shape[0]);
        if U is None:
            return ys;
        else:
            return ys[T.arange(U.shape[0]), U];
    
    def sample_action(self, state, return_prob = False):
        try:
            prob = self._compute_prob(state, self.mem); #based on the current mem and this new state, predict prob.
        except Exception:
            embed() 
        if math.isnan(prob[0]):
            prob = np.ones(prob.shape[0]) / (prob.shape[0]*1.); 
        self.update_mem(x = state); #update the mem to include the new state. 
        xk = np.arange(self.output_dim);
        custm = stats.rv_discrete(name='custm', values=(xk, prob));
        action = custm.rvs();
        if return_prob is True:
            return action, prob[action];
        else:
            return action;



if __name__ == '__main__':
    
    pi = RNN_Discrete_Policy(x_dim = 4, output_dim = 2, nh = 16, params = None);
    
    state = np.random.rand(4);
    p = pi.sample_action(state, return_prob = True);

    #X = np.random.rand(10,4);
    #U = np.ones(10)#.astype('int');
    #rr = pi._test_prob(X,U);
        
    