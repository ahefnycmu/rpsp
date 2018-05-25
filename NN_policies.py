'''
Policies and policy updaters for neural network-based policies
'''

from policies import BasePolicy
import numpy as np
import theano
import theano.tensor as T
from scipy import stats
import math
from models import ObservableModel
from envs.environments import GymEnvironment
from envs.environments import PartiallyObservableEnvironment
from SGD_opt import adam
from SGD_opt import sgd
from IPython import embed
from policy_learn import BasePolicyUpdater
from policy_learn import learn_policy
from policy_learn import learn_model_policy
from envs.simulators import CartpoleContinuousSimulator
from envs.environments import ContinuousEnvironment
from PSIM.kernel_ridge_regression import RidgeRegression
from psr_lite.psr_lite.utils.nn import dbg_print_shape, dbg_print_stats
import nn_policy_updaters

#########################Single Layer class #########################
class Layer(object):
    '''
    x --> Linear --> Non-Linear (ReLu) -->output.
    y = sigma or Relu or Tanh (Layer_norm(Wx + b))
    '''
    def __init__(self, d_in, d_out, layer_id = 0, activation='relu', rng=None):
        self.d_in = d_in;
        self.d_out = d_out;
        #self.nh = nh;
        self.rng = rng
        self.layer_id = layer_id;
        name_W = 'layer_id_{}_W'.format(layer_id);
        name_b = 'layer_id_{}_b'.format(layer_id);
        self.W = theano.shared(name = name_W, 
                value = np.sqrt(2./(self.d_in + self.d_out))
                *rng.randn(self.d_out, self.d_in).astype(theano.config.floatX));
        self.W_s = T.matrix('non_shared_W');
        
        self.b = theano.shared(name = name_b, 
                    value = np.zeros((self.d_out)).astype(theano.config.floatX));
        self.b_s = T.vector('non_shared_b');

        self.params = [self.W, self.b];
        self.params_nonshared = [self.W_s, self.b_s];
                                
        if activation == 'relu':
            self._activation = lambda x: T.nnet.relu(x, alpha = 0.01)
        elif activation == 'tanh':
            self._activation = T.nnet.tanh
        else:
            self._activation = activation

    def _minibatch_forward(self, X):
        #X = dbg_print_shape("Xinlayer.shape=", X)
        #self.W = dbg_print_shape("Wlayer.shape=", self.W)
        lin = T.dot(X, self.W.T)
        lin.name ='minibatchLayerfwd::NNpolicies'
        Linear_output = lin + self.b; #num of samples X d_out.       
        non_linear_output = self._activation(Linear_output)   
        return non_linear_output; 
    
    def _minibatch_forward_nonshared(self, X):
        lin = T.dot(X, self.W_s.T)
        lin.name ='minibatchLayerfwd_ns::NNpolicies'
        Linear_output = lin + self.b_s; #num of samples X d_out. 
        non_linear_output = self._activation(Linear_output)       
        return non_linear_output; 
    
    def _minibatch_forward_nonshared_single(self, x):
        lin = T.dot(self.W_s, x)
        lin.name ='minibatchLayerfwd_nss::NNpolicies'
        Linear_output = lin + self.b_s; #num of samples X d_out. 
        non_linear_output = self._activation(Linear_output)  
        return non_linear_output;

######################################################################

'''
Feed-forward NN Policies use a feed forward network to map a state to an action
'''        
class Policy(BasePolicy):
    def __init__(self, x_dim, output_dim, num_layers, nh, activation='relu', params = None, rng=None):
        self.x_dim = x_dim; #the dimension of the input of the policy.
        self.output_dim = output_dim; #(1) the number of actions (discrete) or the dimension of the action (continuous)
        self.num_layers = num_layers;
        self.nh = nh;
        self.params = [];
        self.params_non_shared = [];
        self.layers= [];
        self.rng = rng
        #construction:
        dim_in = self.x_dim;
        for i in xrange(self.num_layers):
            layer = Layer(dim_in, nh[i], i, activation=activation, rng=self.rng);
            dim_in = nh[i]; 
            self.params = self.params + layer.params;
            self.params_non_shared = self.params_non_shared + layer.params_nonshared;
            self.layers.append(layer);
        
        self.W_y = theano.shared(name = 'W_y', value = np.sqrt(2./(dim_in+self.output_dim))
                *np.zeros((self.output_dim, dim_in)).astype(theano.config.floatX));
        self.W_y_s = T.matrix('non_shared_W_y');
        self.b_y = theano.shared(name = 'b_y', 
                    value = np.zeros(self.output_dim).astype(theano.config.floatX));
        self.b_y_s = T.vector('non_shared_b_y');
        self.params.append(self.W_y);
        self.params.append(self.b_y);
        self.params_non_shared.append(self.W_y_s);
        self.params_non_shared.append(self.b_y_s);
        #if initialization is available: 
        if params is not None:
            assert len(params) == len(self.params);
            for i in xrange(0, len(params)):
                assert params[i].shape == self.params[i].get_value().shape;
                self.params[i].set_value(params[i]);

        self._t_X = T.matrix('X');
        
    def _save(self):
        params = dict([(p.name,p.get_value()) for p in self.params])
        return params
        
        
    def _load(self, params):
        if len(params)>0:
            weight=1.0
            print 'load policy new way'
            for p in self.params:
                p_shape = params[p.name].shape
                t_shape = p.shape.eval().tolist()
                if len(p_shape)==1:
                    pval = self.rng.randn(t_shape[0]) * np.sqrt(2.0/float(np.prod(t_shape)))*weight
                    pval[-p_shape[0]:] = params[p.name]
                    p.set_value(pval)
                elif len(p_shape)==2:
                    pval = self.rng.randn(t_shape[0],t_shape[1]) * np.sqrt(2.0/float(np.prod(t_shape)))*weight
                    pval[-p_shape[0]:,-p_shape[1]:] = params[p.name]
                    p.set_value(pval)
                print p.name, p.get_value().sum()
            self.reset()
        return
        
    @property
    def reactive_policy(self):
        return self

    def get_params(self):
        flatten_grad = np.concatenate([p.get_value().ravel() for p in self.params],axis = 0);
        return flatten_grad;

    def set_params(self, params):
        assert params.shape[0] == self.parameter_dim;
        start = 0;
        if np.isnan(params).any() or np.isinf(params).any():
            print 'param is nan rffpsr policy! not updated'
            return
        for i in xrange(len(self.params)):
            length = (self.params[i].get_value().ravel()).shape[0];
            self.params[i].set_value(params[start:start+length].reshape(self.params[i].get_value().shape));
            start = start + length;


    def reset(self):
        pass;
        
    def project(self, proj):
        dim = proj.shape[1]
        L0 = self.layers[0]
        L0.d_in = dim
        self.x_dim = dim
        
        #project weight matrix
        W0 = self.params[0]
        W0_new = np.dot(W0.get_value(), proj)
        W0.set_value(W0_new)
        return
        

    def compute_action_prob(self, state, action):
        raise NotImplementedError;

    def compute_gradient_ll_wrt_state(self, states, actions):
        pass;

class DiscretePolicy(Policy):
    def __init__(self, x_dim, output_dim, num_layers, nh, params = None, rng=None):
        Policy.__init__(self, x_dim, output_dim, num_layers, nh, params, rng=rng);
        self.discrete = True;

        self._prob_vecs = self._t_compute_prob(self._t_X)
        self._compute_probs = theano.function(inputs = [self._t_X], 
                                    outputs = self._prob_vecs,
                                    allow_input_downcast=True);
        tmp_flatten_grad = np.concatenate([p.get_value().ravel() for p in self.params],axis = 0);
        self.parameter_dim = tmp_flatten_grad.shape[0];
        
        X = T.matrix();
        U = T.ivector();
        self._t_construct_gradient_computation(X, U);

    ################
    def _t_compute_prob_nonshared_single(self, x, u):
        input_x = x;
        output = x;
        for i in xrange(len(self.layers)):
            output = self.layers[i]._minibatch_forward_nonshared(input_x);
            input_x = output;
        pred = T.dot(self.W_y_s, output) + self.b_y_s;
        prob_vec = T.nnet.softmax(pred)[0];
        #return prob_vecs[T.arange(U.shape[0]), T.argmax(U,axis = 1)];
        return prob_vec[u];
    
    def _t_compute_gradient_wrt_x_single(self, x, u):
        log_prob = T.log(self._t_compute_prob_nonshared_single(x = x, u = u));
        grad = T.grad(cost = log_prob, wrt = x);
        return grad;
    
    def _t_compute_gradient_wrt_X(self, X, U):
        grads,_ = theano.scan(fn = self._t_compute_gradient_wrt_x_single, 
                            sequences = [X, U], n_steps=X.shape[0]);
        return grads;
    
    def _t_construct_gradient_computation(self, X, U):
        givens = {};
        for i in xrange(len(self.params)):
            givens.update({self.params_non_shared[i]:self.params[i]});

        grads = self._t_compute_gradient_wrt_X(X, U);
        self.fn_compute_gradient_wrt_X = theano.function(inputs = [X, U], outputs = grads, 
                                    givens = givens, allow_input_downcast=True); 
        
    def compute_gradient_ll_wrt_state(self, states, actions):
        grads = self.fn_compute_gradient_wrt_X(states, actions);    
        return grads;       

    ###########################
    def _t_compute_prob_single(self, x):
        Y = self._t_compute_prob(x.reshape((1,-1)))
        return Y[0]
    
    def _t_compute_prob(self, X, U = None): #U is a two 2d matrix, each row is one-hot encoded.
        input_X = X;
        output = X;
        for i in xrange(len(self.layers)):
            output = self.layers[i]._minibatch_forward(input_X);
            input_X = output;
        pred = T.dot(output, self.W_y.T) + self.b_y;
        prob_vecs = T.nnet.softmax(pred);
        if U is None:
            return prob_vecs;
        else:
            #return prob_vecs[T.arange(U.shape[0]), T.argmax(U,axis = 1)];
            return prob_vecs[T.arange(U.shape[0]), U];  

    def sample_action(self, state):
        #assert state.ndim == 1;
        
        prob = self._compute_probs(state.reshape(1,-1))[0];
        if math.isnan(prob[0]):
            print "generated a nan probability..."
            embed();

        xk = np.arange(self.output_dim);
        custm = stats.rv_discrete(name='custm', values=(xk, prob));
        action = custm.rvs();
        
        return action, prob[action], {}        
    
    def compute_action_prob(self, state, action):
        prob = self._compute_probs(state.reshape(1,-1))[0];
        #print prob
        return prob[action]; #action is from [0, output_dim].

class ContinuousPolicy(Policy):
    def __init__(self,x_dim=None, output_dim=None, num_layers=None, nh=None, activation='relu', params = None, rng=None, min_std=0.0):
        Policy.__init__(self, x_dim, output_dim, num_layers, nh, activation=activation, params = None,\
                        rng=rng);
        self.discrete = False;
        self._min_std = min_std

        #diagnoal variance parameters:
        self.r = theano.shared(name = 'diag params', 
                value = -np.zeros(self.output_dim).astype(theano.config.floatX)); #this initliazation makes sure that the std (exp(r)) is around 0.3, so that 3-std is around 1, which is the max control input for mujoco setups. 
        self.r_s = T.vector('r_non_shared');
        self.params.append(self.r); 
        self.params_non_shared.append(self.r_s);

        tmp_flatten_grad = np.concatenate([p.eval().ravel() for p in self.params],axis = 0);
        self.parameter_dim = tmp_flatten_grad.shape[0];

        self._t_diag_vars = (self._t_std())**2;
        #self._t_diag_vars = dbg_print_stats('act_var', self._t_diag_vars)
        self._t_pred_means = self._t_compute_mean(self._t_X); 
        self.construct_normal_dis = theano.function(inputs = [self._t_X],
                outputs = [self._t_pred_means, self._t_diag_vars], 
                allow_input_downcast=True);

        self._t_U = T.matrix('U');
        probs = self._t_compute_prob(self._t_X, self._t_U);
        self._compute_prob = theano.function(inputs = [self._t_X, self._t_U],
                                outputs = probs,
                                allow_input_downcast=True);

        self._t_construct_gradient_computation(self._t_X, self._t_U);

    def project(self, proj):
        Policy.project(self, proj)
        tmp_flatten_grad = np.concatenate([p.get_value().ravel() for p in self.params],axis = 0);
        self.parameter_dim = tmp_flatten_grad.shape[0];
        return

    ################################
    def _t_compute_prob_nonshared_single(self, x, u):
        variances = (self._t_std())**2 ; #std->var. 
        det = T.prod(variances);
        coeff = 1./T.power(T.power(2.*np.pi, self.output_dim)*det, 0.5);
        input_x = x;
        output = x;
        for i in xrange(len(self.layers)):
            output = self.layers[i]._minibatch_forward_nonshared_single(input_x);
            input_x = output;
        u_means = T.dot(self.W_y_s, output) + self.b_y_s;
        du = u - u_means;
        m_dis = 0.5*(1./variances * du).dot(du);
        prob = coeff * T.exp(-m_dis);
        return prob;

    def _t_compute_gradient_wrt_x_single(self, x, u):
        log_prob = T.log(self._t_compute_prob_nonshared_single(x = x, u = u));
        grad = T.grad(cost = log_prob, wrt = x);
        return grad;
    
    def _t_compute_gradient_wrt_X(self, X, U):
        grads,_ = theano.scan(fn = self._t_compute_gradient_wrt_x_single, 
                            sequences = [X, U], n_steps=X.shape[0]);
        return grads; 
    
    def _t_construct_gradient_computation(self, X, U):
        givens = {};
        for i in xrange(len(self.params)):
            givens.update({self.params_non_shared[i]:self.params[i]});

        grads = self._t_compute_gradient_wrt_X(X, U);
        self.fn_compute_gradient_wrt_X = theano.function(inputs = [X, U], outputs = grads, 
                                    givens = givens, allow_input_downcast=True); 

    def compute_gradient_ll_wrt_state(self, states, actions):
        #print states.shape
        grads = self.fn_compute_gradient_wrt_X(states, actions);    
        return grads;       
    
    
    ################################
    def _t_std(self):
        #compute diag variance:
        std = T.exp(self.r) + self._min_std
        return std
    
    
    def _t_compute_mean(self,X):
        input_X = X;
        output = X;
        for i in xrange(len(self.layers)):
            output = self.layers[i]._minibatch_forward(input_X);
            input_X = output;
        pred = T.dot(output, self.W_y.T) + self.b_y;
        return pred;
    
    def _t_compute_single_m_dis(self, du, variances):
        #assert du.shape[0] == variances.shape[0];
        return 0.5*(1./variances * du).dot(du);

    def _t_compute_gaussian(self, X):        
        return self._t_compute_mean(X), T.tile(T.log(self._t_std()), (X.shape[0], 1))

    def _t_compute_prob(self, X, U):
        std = self._t_std()
        variances = std**2     
        det = T.prod(variances)
        coeff = 1./T.power(T.power(2.*np.pi, self.output_dim)*det, 0.5)
        U_means = self._t_compute_mean(X)
        #U_means = dbg_print_stats('Umeans', U_means)                              
        
        d_U = (U - U_means) / std
        #m_diss = 0.5 * T.sum(d_U * d_U, axis=1)
        m_diss = 0.5 * T.sum(d_U * d_U, axis=1)
        probs = coeff * T.exp(-m_diss);
        return probs+1e-300; #a 1-d vector: each element represents pi(u_i | s_i).
    
    def sample_action(self, state):
        #assert state.ndim == 1
        [means, diag_vars] = self.construct_normal_dis(state.reshape(1,-1));
        #print('diag_vars', diag_vars)
        mean = means[0];
        diag_var = diag_vars #[0];
        
        assert mean.shape == (self.output_dim,)
        assert diag_var.shape == (self.output_dim,)

        action = self.rng.randn(self.output_dim) * (diag_var**0.5) + mean;
        assert (diag_vars>1e-10).all(), 'Diag Variance is non positive. [NN_policies]'
        #clip:
        action_c = np.copy(action);
        #for i in xrange(0,len(action)):
        #    if action_c[i] > 1.:
        #        action_c[i] = 1.;
        #    elif action_c[i] < -1:
        #        action_c[i] = -1.;

        
        prob = self._compute_prob(state.reshape(1,-1), action.reshape(1,-1))[0]

        diagnostics = {'act_var' : diag_var}
        return action_c, prob, diagnostics
    
    def compute_action_prob(self, state, action):
        #pass;
        prob = self._compute_prob(state.reshape(1,-1), action.reshape(1,-1))[0];
        return prob;
    
class ContinuousExplorationPolicy(ContinuousPolicy):
    def __init__(self, exp_strategy, *args, **kwargs):
        super(ContinuousExplorationPolicy, self).__init__(*args, **kwargs)
        self._exploration_strategy = exp_strategy(self)
        #exp_strategy._base_policy = self
        self._iter=0
    
    def sample_action(self, state):
        out = super(ContinuousExplorationPolicy,self).sample_action(state)
        act, act_prob, act_info = self._exploration_strategy.get_action(self._iter, out)
        self._iter+=1.0
        return act, act_prob, act_info
        
 
#############Variance Reduced Reinforce############################3
class VR_Reinforce_PolicyUpdater(BasePolicyUpdater): #using Rich Sutton's policy gradient theorem.
# a variance-reduced reinforce algorithm:
    def __init__(self, policy, max_traj_length, num_trajs, lr = 1e-2, gamma = 0.99, baseline = True, **kwargs):
        self._policy = policy;
        self.baseline = baseline;
        self.gamma = gamma;
        self.gamma_seq = np.array([self.gamma**(i) for i in xrange(10000)]); 

        if self.baseline is True:
            self.linear_reg = RidgeRegression(Ridge = 1e-7);
            self.states_set = None;
            self.ctgs_set = None;
            self.max_num = 50000;

        self._t_X = T.matrix('state');
        if self._policy.discrete == True:
            self._t_U = T.ivector('control');
        else:
            self._t_U = T.matrix('control');
        self._t_ctg = T.vector('ctg');

        self._t_avg_ctg = self._t_cost(self._t_X, self._t_U, self._t_ctg);
        self._t_lr = T.scalar('lr');
        self._t_update = adam(self._t_avg_ctg, 
                         self._policy.params, self._t_lr);
        
        self.gradient_descent = theano.function(
                        inputs = [self._t_X, self._t_U,
                        self._t_ctg, self._t_lr], outputs = self._t_avg_ctg,
                        updates=self._t_update,
                        allow_input_downcast=True);
        self.lr = lr;
        
    
    @property
    def policy(self):
        return self._policy;

    def _t_cost(self, X, U, Ctg):
        probs = self._policy._t_compute_prob(X, U);
        cost = T.mean(T.log(probs) * Ctg);
        return cost;

    def _transfer_trajs_to_matrixformat(self, trajs):
        num_trajs = len(trajs);
        total_costs = [];
        total_costs.append(-np.sum(trajs[0][3]));
        states = trajs[0][1][:-1];
        actions = trajs[0][2][1:];
        ctgs = np.array([-np.sum(trajs[0][3][i:]) for i in range(1,trajs[0][3].shape[0])]);
        for i in xrange(0, num_trajs):
            states = np.concatenate([states, trajs[i][1][:-1]], axis = 0);
            actions= np.concatenate([actions,trajs[i][2][1:]], axis = 0);
            tmp_ctgs = np.array([-np.sum(trajs[i][3][j:]*self.gamma_seq[0:len(trajs[i][3][j:])]) for j in range(1,trajs[i][3].shape[0])]);
        
            ctgs = np.concatenate([ctgs, tmp_ctgs]);
            total_costs.append(-np.sum(trajs[i][3]));
        
            for k in xrange(trajs[i].policy_grads.shape[0]):
                trajs[i].policy_grads[k,:] *= (-np.sum(trajs[i][3][k:])); 

        if self.baseline is True:
            #add baseline. 
            self.states_set = states if self.states_set is None else np.concatenate((self.states_set,states),axis=0);
            self.ctgs_set = ctgs if self.ctgs_set is None else np.concatenate((self.ctgs_set,ctgs));
            assert self.states_set.shape[0] == self.ctgs_set.shape[0];
            if self.states_set.shape[0] > self.max_num:
                self.states_set = self.states_set[-self.max_num:];
                self.ctgs_set = self.ctgs_set[-self.max_num:];
            
            self.linear_reg.fit(input_X = self.states_set, Y = self.ctgs_set.reshape(-1,1));
            pred_ctgs = self.linear_reg.predict(input_X = states);
            ctgs = ctgs - pred_ctgs.flatten();

        if self._policy.discrete:
            return [states, actions.reshape(actions.shape[0]), ctgs, total_costs];
            #return [states, np.argmax(actions,axis = 1), ctgs, total_costs];
        else:
            return [states, actions, ctgs, total_costs]; 
    
    def update(self, trajs):
        for i in xrange(len(trajs)):
            if self._policy.discrete is False:
                trajs[i].policy_grads = self._policy.compute_gradient_ll_wrt_state(states = trajs[i].states,actions=trajs[i].act);
            elif self._policy.discrete is True:
                trajs[i].policy_grads =self._policy.compute_gradient_ll_wrt_state(states = trajs[i].states, actions = trajs[i].act.reshape(trajs[i].act.shape[0]));
        [states, actions, ctgs, tcs] = self._transfer_trajs_to_matrixformat(trajs);
        
        output = self.gradient_descent(states,actions, ctgs, self.lr);
    
        return [np.mean(tcs), np.std(tcs)];
############################End of Variance-Reduced Reinforce#######################
        
'''
RNN Policy uses RNN as both a state model and an action function.
This policy is typically used with ObservableModel as a model.
'''
class RNN_Policy(BasePolicy):
    def __init__(self, x_dim, a_dim, output_dim, nh, LSTM = False, params = None, rng=None, ext=False):
        self.x_dim = x_dim; 
        self.a_dim = a_dim;
        self.output_dim = output_dim;
        self.nh = nh;
        self.params = [];
        self.rng = rng
        self.LSTM = LSTM;
        self.ext_dim = self.nh if ext else 0
        if LSTM is False:
            self.W_x=theano.shared(name='W_x',value= np.sqrt(2./(self.x_dim+self.output_dim+self.nh))
                        *self.rng.randn(self.nh,self.x_dim+self.output_dim).astype(theano.config.floatX));
            self.W_h=theano.shared(name='W_h',value= np.sqrt(2./(self.nh+self.nh))
                        *self.rng.randn(self.nh,self.nh).astype(theano.config.floatX));
            self.hb = theano.shared(name='hn',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
            self.h_0 = theano.shared(name='h0',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
        
            self.W = theano.shared(name='W',value=np.sqrt(2./(self.output_dim+self.nh))
                        *self.rng.randn(self.output_dim,self.nh).astype(theano.config.floatX));
        
            self.b = theano.shared(name='b',
                        value=np.zeros(self.output_dim).astype(theano.config.floatX));
        
            self.params = [self.W_x,self.W_h,self.hb,self.h_0,self.W,self.b];
        else:
            self.W_i = theano.shared(name='W_i',value= np.sqrt(2./(self.x_dim+self.output_dim+2*self.nh))
                        *self.rng.randn(self.nh,self.nh+self.x_dim+self.output_dim+self.ext_dim).astype(theano.config.floatX));
            self.b_i = theano.shared(name='b_i',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
            self.W_f = theano.shared(name='W_f',value= np.sqrt(2./(self.x_dim+self.output_dim+2*self.nh))
                        *self.rng.randn(self.nh,self.nh+self.x_dim+self.output_dim+self.ext_dim).astype(theano.config.floatX));
            self.b_f = theano.shared(name='b_f',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
            self.W_o = theano.shared(name='W_o',value= np.sqrt(2./(self.x_dim+self.output_dim+2*self.nh))
                        *self.rng.randn(self.nh,self.nh+self.x_dim+self.output_dim+self.ext_dim).astype(theano.config.floatX));
            self.b_o = theano.shared(name='b_o',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
            self.W_c = theano.shared(name='W_c',value= np.sqrt(2./(self.x_dim+self.output_dim+2*self.nh))
                        *self.rng.randn(self.nh,self.nh+self.x_dim+self.output_dim).astype(theano.config.floatX));
            self.b_c = theano.shared(name='b_f',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
            self.h_0 = theano.shared(name='h_0',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
            self.c_0 = theano.shared(name='c_0',
                        value=np.zeros(self.nh).astype(theano.config.floatX));
            self.W = theano.shared(name='W',value=np.sqrt(2./(self.output_dim+2*self.nh))
                        *self.rng.randn(self.output_dim,2*self.nh).astype(theano.config.floatX));
            self.b = theano.shared(name='b',
                        value=np.zeros(self.output_dim).astype(theano.config.floatX));
            self.params = [self.W_i,self.b_i,self.W_f,self.b_f,self.W_c,self.b_c,
                        self.W_o,self.b_o,self.h_0,self.c_0,self.W,self.b];
                
        if params is not None:
            assert len(params) == len(self.params);
            for i in xrange(len(params)):
                self.params[i].set_value(params[i]);
    def _save(self):
        params = {}
        params['all'] = self.get_params()
        return params
        
        
    def _load(self, params):
        print 'load RNN policy'
        self.set_params(params['all'])
        self.reset()
        return
#Continuous RNN policy
class RNN_Continuous_Policy(RNN_Policy):
    def __init__(self, x_dim, a_dim, output_dim, nh, LSTM = False, params = None, rng=None, ext=False):
        RNN_Policy.__init__(self,x_dim,a_dim, output_dim,nh,LSTM, params=params, rng=rng, ext=ext);
        self.r = theano.shared(name = 'diag params', 
            value = -np.zeros(self.output_dim).astype(theano.config.floatX)); 
        self.params.append(self.r); 
        self._init_a = np.zeros(self.output_dim);
        self.LSTM = LSTM;
        self._t_single_step_LSTM_forward = self._t_single_step_extLSTM_forward if ext else self._t_single_step_tradLSTM_forward

        self.discrete = False;
        self._t_mem = T.vector('mem');
        self._t_state = T.vector('state');
        self._t_a = T.vector('a');
        self._t_cell = T.vector('cell');
        self._t_diag_vars = (T.exp(self.r))**2;

        if self.LSTM is False:
            [self._t_updated_mem, self._t_mean] = self._t_single_step_forward(self._t_state,self._t_a, self._t_mem);
            self._update_mem=theano.function(inputs = [self._t_state,self._t_a, self._t_mem],
                                    outputs=self._t_updated_mem, allow_input_downcast=True);
            t_mean = T.dot(T.concatenate((self._t_mem,self._t_state),axis=0), self.W.T) + self.b;
            self.construct_normal_dis = theano.function(inputs = [self._t_state,self._t_mem],
                    outputs = [t_mean, self._t_diag_vars], 
                    allow_input_downcast=True);
        
        elif self.LSTM is True:
            [self._t_updated_mem,self._t_updated_cell]= self._t_single_step_LSTM_forward(self._t_state,self._t_a,
                                    self._t_mem,self._t_cell)[0:2];
            self._update_mem=theano.function(inputs = [self._t_a,self._t_state,self._t_mem,self._t_cell],
                                    outputs=self._t_updated_mem, allow_input_downcast=True);
            self._update_cell=theano.function(inputs=[self._t_a,self._t_state,self._t_mem,self._t_cell],
                                    outputs=self._t_updated_cell, allow_input_downcast=True);
            t_mean = T.dot(T.concatenate((self._t_mem,self._t_cell),axis=0),self.W.T) + self.b;
            self.construct_normal_dis = theano.function(inputs=[self._t_mem,self._t_cell],
                        outputs = [t_mean,self._t_diag_vars],
                        allow_input_downcast=True);

        self.reset();

    def _t_compute_single_m_dis(self, du, variances):
        #assert du.shape[0] == variances.shape[0];
        return 0.5*(1./variances * du).dot(du);

    def reset(self):
        self.mem = None;
        self.cell = None;
        #self.mem = self.h0.get_value();
        self.act = self._init_a;
        #self.update_mem(x);
        return self.mem;

    def update_mem(self, x):
        
        assert not np.isnan(x).any(), 'not nan x in update mem RNN policy'
        if self.mem is None:
            self.mem = self.h_0.get_value();
            if np.isnan(self.mem).any():
                print 'not nan h0 in update mem RNN policy'
                embed()
            
            if self.LSTM is True:
                self.cell = self.c_0.get_value();
        else:
            if self.LSTM is False:
                self.mem = self._update_mem(self.act, x, self.mem);  
            else:
                self.mem = self._update_mem(self.act,x,self.mem,self.cell);
                self.cell = self._update_cell(self.act,x,self.mem,self.cell);
#        assert not 
        if np.isnan(self.mem).any():
            print 'not nan mem in update mem RNN policy'
            embed()
        if self.LSTM is True:
            if np.isnan(self.cell).any():
                print 'not nan cell in update mem RNN policy'
                embed()
        return

    def _t_single_step_forward(self, x, a, h):
        y = T.dot(T.concatenate((h,x),axis=0),self.W.T) + self.b;
        #h_tp1 = T.nnet.tanh(T.dot(h,self.W_h.T)+T.dot(T.concatenate((x,a),axis = 0),self.W_x.T)+self.hb, alpha = 0.01);
        h_tp1 = T.nnet.relu(T.dot(h,self.W_h.T)+T.dot(T.concatenate((x,a),axis = 0),self.W_x.T)+self.hb, alpha = 0.01);
        return [h_tp1, y]; #return the mean. 

    def _t_single_step_tradLSTM_forward(self, x, a, h, c):
        f = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a),axis=0),self.W_f.T) + self.b_f)#, alpha = 0.01);
        i = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a),axis=0),self.W_i.T) + self.b_i)#, alpha = 0.01);
        tilde_c =   T.tanh(T.dot(T.concatenate((h,x,a),axis=0),self.W_c.T) + self.b_c);
        c_new = f*c + i*tilde_c;
        o = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a),axis=0),self.W_o.T) + self.b_o)#, alpha = 0.01);
        h_new = o * T.tanh(c_new);
        y = T.dot(T.concatenate((h,c),axis=0),self.W.T) + self.b;
        #y = dbg_print_stats('y_single_step',y)
        #h_new = dbg_print_stats('hnew_single_step',h_new)
        #c_new = dbg_print_stats('hnew_single_step',c_new)
        return [h_new, c_new, y];
    
    def _t_single_step_extLSTM_forward(self, x, a, h, c):
        f = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a,c),axis=0),self.W_f.T) + self.b_f)#, alpha = 0.01);
        i = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a,c),axis=0),self.W_i.T) + self.b_i)#, alpha = 0.01);
        tilde_c =   T.tanh(T.dot(T.concatenate((h,x,a),axis=0),self.W_c.T) + self.b_c);
        c_new = f*c + i*tilde_c;
        o = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a,c),axis=0),self.W_o.T) + self.b_o)#, alpha = 0.01);
        h_new = o * T.tanh(c_new);
        y = T.dot(T.concatenate((h,c),axis=0),self.W.T) + self.b;
        #y = dbg_print_stats('y_single_step',y)
        #h_new = dbg_print_stats('hnew_single_step',h_new)
        #c_new = dbg_print_stats('hnew_single_step',c_new)
        return [h_new, c_new, y];
    
    def _t_compute_prob(self, X, U):
        if self.LSTM is False:
            [hs,ys],_ = theano.scan(fn=self._t_single_step_forward,
                                sequences=[X,U], outputs_info=[self.h_0,None],
                                n_steps = X.shape[0]);
        else:
            #self.W = dbg_print_stats('W', self.W)
            [hs,cs,ys],_ = theano.scan(fn=self._t_single_step_LSTM_forward, 
                                sequences =[X,U], outputs_info=[self.h_0,self.c_0,None],
                                n_steps = X.shape[0]);

        variances = (T.exp(self.r))**2; #std->var. 
        #variances = dbg_print_stats('var_prob', variances)
        det = T.prod(variances);
        coeff = 1./T.power(T.power(2.*np.pi, self.output_dim)*det, 0.5);
        U_means = ys; #using prestates but should we  use prestates or post states?
         
        #U_means = dbg_print_stats('umeans_prob', U_means)
        d_U = U - U_means;# aligned??
        m_diss,_ = theano.scan(fn = self._t_compute_single_m_dis, 
                    sequences = [d_U], non_sequences=variances, n_steps=d_U.shape[0]);
        
        #m_diss = dbg_print_stats('mdiss_prob', m_diss)
        probs = coeff * T.exp(-m_diss) + 1e-300;
        
        #probs = dbg_print_stats('prob_prob', probs)
        return probs;

    def _compute_prob_normal(self, state, mean, diag_vars):
        det = np.prod(diag_vars);
        coeff = 1./np.power(np.power(2.*np.pi, self.output_dim)*det, 0.5);
        dd = state - mean;
        prob = coeff * np.exp(-0.5*(1./diag_vars * dd).dot(dd));
        #print 'prob ', prob
        return prob+1e-100;

    def sample_action(self, state):
        self.update_mem(x = state);
        if self.LSTM is False:
            [mean, diag_vars] = self.construct_normal_dis(state,self.mem);
        else:
            [mean, diag_vars] = self.construct_normal_dis(self.mem,self.cell);
            
        action = self.rng.randn(self.output_dim) * (diag_vars**0.5) + mean;
        self.act = action;
        #print 'mean:',mean
        #print 'std:',diag_vars
        #print 'act:',self.act
        if np.isnan(action).any():
            print "nan action"
            embed()
                        
        prob = self._compute_prob_normal(action, mean, diag_vars);
        return action, prob, {};        
    
#Discrete RNN-Policy
class RNN_Discrete_Policy(RNN_Policy):
    def __init__(self, x_dim, a_dim, output_dim, nh, LSTM = False, params = None, rng=None):
        #a_dim = 0;
        RNN_Policy.__init__(self, x_dim, a_dim, output_dim, nh, LSTM, params,rng=rng);
        #self._t_init_a = theano.shared(name = 'void action', 
        #    value = np.zeros(self.a_dim).astype(theano.config.floatX)); 
        self._init_a = np.zeros(self.output_dim);
        self.discrete = True;
        self._t_mem = T.vector('mem');
        self._t_state = T.vector('mem');
        self._t_a = T.vector('a');
        self._t_cell = T.vector('cell');

        if self.LSTM is False: #simple RNN:
            self._t_updated_mem= self._t_single_step_forward(self._t_state,self._t_a,self._t_mem)[0];
            self._update_mem=theano.function(inputs = [self._t_a,self._t_state,self._t_mem],
                                    outputs=self._t_updated_mem, allow_input_downcast=True);
            t_prob = T.nnet.softmax(T.dot(self._t_mem,self.W.T) + self.b)[0];
            self._compute_prob=theano.function(inputs = [self._t_mem],
                                    outputs=t_prob, allow_input_downcast=True);
        else: #LSTM update
            [self._t_updated_mem,self._t_updated_cell]= self._t_single_step_LSTM_forward(self._t_state,self._t_a,
                                    self._t_mem,self._t_cell)[0:2];
            self._update_mem=theano.function(inputs = [self._t_a,self._t_state,self._t_mem,self._t_cell],
                                    outputs=self._t_updated_mem, allow_input_downcast=True);
            self._update_cell=theano.function(inputs=[self._t_a,self._t_state,self._t_mem,self._t_cell],
                                    outputs=self._t_updated_cell, allow_input_downcast=True);
            t_prob = T.nnet.softmax(T.dot(T.concatenate((self._t_mem,self._t_cell),
                        axis=0),self.W.T) + self.b)[0];
            self._compute_prob=theano.function(inputs = [self._t_mem,self._t_cell],
                                    outputs=t_prob, allow_input_downcast=True);
        
        
        self.reset();

    def reset(self):
        self.mem = None;
        self.cell = None;
        #self.mem = self.h0.get_value();
        self.act = self._init_a;
        #self.update_mem(x);
        return self.mem;

    def update_mem(self, x):
        if self.mem is None:
            #assert not np.isnan(self.h_0.get_value()).any(), 'not nan mem in update mem RNN policy'
            if np.isnan(self.h_0.get_value()).any():
                print 'not nan mem in update mem RNN policy'
            self.mem = self.h_0.get_value();
            if self.LSTM is True:
                self.cell = self.c_0.get_value();
        else:
            if self.LSTM is False:
                self.mem = self._update_mem(self.act, x, self.mem);  
            else:
                self.mem = self._update_mem(self.act,x,self.mem,self.cell);
                self.cell = self._update_cell(self.act,x,self.mem,self.cell);
        if np.isnan(self.mem).any():
            print 'not nan mem in update mem RNN policy'
            

    def _t_single_step_forward(self, x, a, h):
        h_tp1 = T.nnet.relu(T.dot(h,self.W_h.T)+T.dot(T.concatenate((x,a),axis=0),self.W_x.T)+self.hb, alpha = 0.01);
        y = T.nnet.softmax(T.dot(h,self.W.T) + self.b)[0];
        return [h_tp1, y];
    
    def _t_single_step_LSTM_forward(self, x, a, h, c):
        f = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a,c),axis=0),self.W_f.T) + self.b_f)#, alpha = 0.01);
        i = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a,c),axis=0),self.W_i.T) + self.b_i)#, alpha = 0.01);
        tilde_c =   T.tanh(T.dot(T.concatenate((h,x,a),axis=0),self.W_c.T) + self.b_c);
        c_new = f*c + i*tilde_c;
        o = T.nnet.sigmoid(T.dot(T.concatenate((h,x,a,c_new),axis=0),self.W_o.T) + self.b_o)#, alpha = 0.01);
        h_new = o * T.tanh(c_new);
        y = T.nnet.softmax(T.dot(T.concatenate((h,c_new),axis=0),self.W.T)+self.b)[0];
        return [h_new, c_new, y];
        
    ####need to work on this function. 
    def _t_compute_prob(self, X, U): 
        if self.LSTM is False:
            [hs,ys],_ = theano.scan(fn=self._t_single_step_forward,
                                sequences=[X,U], outputs_info=[self.h_0,None],
                                n_steps = X.shape[0]);
        else:
            [hs,cs,ys],_ =theano.scan(fn = self._t_single_step_LSTM_forward,
                                sequences=[X,U],outputs_info=[self.h_0,self.c_0,None],
                                n_steps=X.shape[0]);

        return T.sum(ys*U, axis=1)
        #return ys[T.arange(U.shape[0]-1), T.argmax(U[1:],axis=1)];
        #U[1:,0]];
    
    def sample_action(self, state):
        #embed()
        assert not np.isnan(self.h_0.get_value()).any(), 'not nan mem in update mem RNN policy'
        self.update_mem(x = state); #incoporate the latest observation and the previous action into the memory.
        if self.LSTM is True:
            prob = self._compute_prob(self.mem,self.cell); #based on the current mem
        else:
            prob = self._compute_prob(self.mem);
        if math.isnan(prob[0]): 
            prob = np.ones(prob.shape[0]) / (prob.shape[0]*1.); 
        
        xk = np.arange(self.output_dim);
        custm = stats.rv_discrete(name='custm', values=(xk, prob));
        action = custm.rvs();
        self.act = np.zeros(self.output_dim);
        self.act[int(action)] = 1.
                 
        return action, prob[action]        

#############################Variance Reduced Reinforce for RNN Policy##############
class VR_Reinforce_RNN_PolicyUpdater(BasePolicyUpdater):
    def __init__(self, policy, max_traj_length, num_trajs, lr = 1e-2, gamma = 0.99, \
                 baseline = True, discount=1.0, clips=[], **kwargs):
        self.max_traj_len = max_traj_length;
        self.num_trajs = num_trajs;
        self._policy = policy;
        self.lr = lr;
        self.clips = clips
        
        self.gamma = gamma;
        self.baseline = baseline;
        if self.baseline is not False:
            self.linear_reg = RidgeRegression(Ridge = 1e-7);
            self.states_set = None;
            self.ctgs_set = None;
            self.max_num = 50000;

        self.gamma_seq = np.array([self.gamma**(i) for i in xrange(10000)]);

        symbolic_list_trajs_X = []; #symbolic representation of a list of trajectories (states)
        symbolic_list_trajs_U = [];#simbolic representation of a list of controls (U)
        symbolic_list_trajs_ctg = []; #simbolic representation of a list of trajectory ctgs. 
        for i in xrange(0, self.num_trajs):
            symbolic_list_trajs_X.append(T.matrix('trajx_{}'.format(i)));
            if self._policy.discrete == True:
                symbolic_list_trajs_U.append(T.imatrix('traju_{}'.format(i)));
            else:
                symbolic_list_trajs_U.append(T.matrix('traju_{}'.format(i)));
            symbolic_list_trajs_ctg.append(T.vector('ctg_{}'.format(i)));
        self._t_list_traj_X = T.stack(symbolic_list_trajs_X);#3d tensor
        self._t_list_traj_U = T.stack(symbolic_list_trajs_U);#2d tensor if discrete, 3d otherwise
        self._t_list_ctg = T.stack(symbolic_list_trajs_ctg);#2d tensor. 

        #self._t_list_traj_X = T.tensor3()
        #self._t_list_ctg = T.matrix()

        self._t_avg_ctg = self._t_cost(self._t_list_traj_X, self._t_list_traj_U, 
                                        self._t_list_ctg);

        #self._test = theano.function(inputs = [self._t_list_traj_X, self._t_list_traj_U,
        #                self._t_list_ctg], outputs = self._t_avg_ctg,
        #                allow_input_downcast=True);

        self._t_lr = T.scalar('lr');
        self._t_update = adam(self._t_avg_ctg, 
                         self._policy.params, self._t_lr, clip_bounds=self.clips);
        
        self.gradient_descent = theano.function(
                        inputs = [self._t_list_traj_X, self._t_list_traj_U,
                        self._t_list_ctg, self._t_lr], outputs = self._t_avg_ctg,
                        updates=self._t_update,
                        allow_input_downcast=True);
                
    @property
    def policy(self):
        return self._policy;

    def _t_single_traj_cost(self, X, U, ctg):
        if self._policy.discrete is True:
            #valid_len = U.shape[0] - T.sum(T.eq(U, -np.inf));
            valid_len = U.shape[0] - T.sum(U[:,0] <= -1);
        else:
            #valid_len = U.shape[0] - T.sum(T.eq(U[:,0],-np.inf));
            valid_len = U.shape[0] - T.sum(U[:,0] <= -1e10);
        
        probs = self._policy._t_compute_prob(X[:valid_len],U[:valid_len]);
        cost = T.mean(T.log(probs)*ctg[:valid_len]);
        return cost; #the average cross this particular trajectory

    def _t_cost(self, Xs, Us, ctgs):
        #Xs: a list of 2d matrix;
        #Us: a list of 1d vector or 2d matrix;
        #ctgs: a list of 1d vector;
        ccs,_ = theano.scan(fn = self._t_single_traj_cost,
                            sequences=[Xs,Us,ctgs], n_steps=Xs.shape[0])#, truncate_gradient=50);
        return T.mean(ccs);

    def _padding(self, trajs): #trajs: obs, state, u, reward.#U is in one-hot encoding.
        assert len(trajs) == self.num_trajs;
        tensor_traj_state = np.zeros((self.num_trajs, 
                        self.max_traj_len, trajs[0][1].shape[1]));
        if self._policy.discrete is True:
            tensor_traj_U = np.ones((self.num_trajs, self.max_traj_len,self._policy.output_dim))*(-np.inf);
        else:
            tensor_traj_U = np.ones((self.num_trajs,self.max_traj_len, 
                        trajs[0][2].shape[1]))*(-np.inf);
        tensor_traj_ctg = np.zeros((self.num_trajs, self.max_traj_len));                
        trajs_tc = [];
        for i in xrange(0, self.num_trajs):
            tensor_traj_state[i,:trajs[i][1].shape[0],:] = trajs[i][1];
            if self._policy.discrete is True:
                #embed()
                #tensor_traj_U[i,1:trajs[i][2].shape[0],:] = trajs[i][2][1:]#.reshape(trajs[i][2][1:].shape[0]);
                #tensor_traj_U[i,0] = self._policy._init_a;
                #tensor_traj_U[i,:trajs[i][2].shape[0],:] = trajs[i][2]
                tensor_traj_U[i,0:trajs[i][2].shape[0],:] = np.zeros((trajs[i][2].shape[0], self._policy.output_dim));
                tensor_traj_U[i,np.arange(trajs[i][2].shape[0]), trajs[i][2][:,0].astype(int)] = 1.;
                #tensor_traj_U[i,:trajs[i][2].shape[0]-1] = np.argmax(trajs[i][2][1:], axis = 1);
            else:
                tensor_traj_U[i,0:trajs[i][2].shape[0]] = trajs[i][2];
                #tensor_traj_U[i,0] = self._policy._init_a;

            tmp_ctgs = np.array([-np.sum(trajs[i][3][j:]*self.gamma_seq[0:len(trajs[i][3][j:])]) for j in range(0,trajs[i][3].shape[0])]);
            tensor_traj_ctg[i,0:tmp_ctgs.shape[0]] = tmp_ctgs;
            trajs_tc.append(-np.sum(trajs[i][3])); #convert reward to cost.

            if self.baseline is not False:
                self.states_set = trajs[i][1] if self.states_set is None else np.concatenate((self.states_set,trajs[i][1]),axis=0);
                self.ctgs_set = tmp_ctgs if self.ctgs_set is None else np.concatenate((self.ctgs_set,tmp_ctgs));
                assert self.states_set.shape[0] == self.ctgs_set.shape[0];
    
                if self.states_set.shape[0] > self.max_num:
                    self.states_set = self.states_set[-self.max_num:];
                    self.ctgs_set = self.ctgs_set[-self.max_num:];

        if self.baseline is not False:
            self.linear_reg.fit(self.states_set, self.ctgs_set.reshape(-1,1));
            for i in xrange(self.num_trajs):
                pred_ctg = self.linear_reg.predict(trajs[i][1]);
                tensor_traj_ctg[i, 0:trajs[i][1].shape[0]] -= pred_ctg.flatten();
        
        return [tensor_traj_state, tensor_traj_U, tensor_traj_ctg, trajs_tc]; 

    def update(self, trajs):
        [list_traj_state,list_traj_U, list_traj_ctg, list_traj_cc] = self._padding(trajs);
        
        output = self.gradient_descent(list_traj_state, list_traj_U,
                                    list_traj_ctg, self.lr);
        return [np.mean(list_traj_cc), np.std(list_traj_cc)];
    
#############################End of VR RNN Reinforce#################################

if __name__ == '__main__':

   # Create and learn a forward NN policy
    #np.random.seed(300);
    rng = np.random.RandomState(300)
    #env = GymEnvironment('CartPole-v0');
    #env = partial_obs_Gym_CartPole_Env('CartPole-v0');
    #env = GymEnvironment('Acrobot-v1');
    #env = GymEnvironment('Swimmer-v1', discrete = False, rng=rng);
    env = GymEnvironment('Swimmer-v1', discrete = False, rng=rng);

    #env = GymEnvironment('HalfCheetah-v1', discrete = False);
    #env = GymEnvironment('Reacher-v1', discrete = False);
    #env = PartiallyObservableEnvironment(GymEnvironment('Swimmer-v1', discrete = False, rng=rng), np.array([0,1,2,3]))
    #env = PartiallyObservableEnvironment(GymEnvironment('Walker2d-v1',discrete = False), np.array([0,1,2,3,4]) )
    #env = PartiallyObservableEnvironment(GymEnvironment('CartPole-v1'),np.array([0,2]));
    #env = PartiallyObservableEnvironment(ContinuousEnvironment('CartPole-v0', CartpoleContinuousSimulator(), rng=rng),np.array([0,2]) );
    #env.reset();
    (x_dim, a_dim) = env.dimensions;
    print x_dim, a_dim;
    output_dim = a_dim #env.action_info[0];
    print output_dim
    #this model could be replace by PSR or PSIM later.
    model = ObservableModel(obs_dim = x_dim); 

    max_traj_length = 500;
    num_trajs = 8;
    
    #pi = DiscretePolicy(x_dim = x_dim, output_dim = output_di    m, num_layers = 1, nh = [16]);
    #pi = ContinuousPolicy(x_dim = x_dim, output_dim = output_dim, num_layers = 1, nh = [64]);
    #embed()
    
    #pi = RNN_Discrete_Policy(x_dim = x_dim, a_dim = a_dim, output_dim=output_dim, nh = 16, LSTM = True);

    #pi = RNN_Continuous_Policy(x_dim=x_dim, a_dim = a_dim, output_dim=output_dim, nh = 64, LSTM = True, rng=rng);
    #pi = ContinuousPolicy(x_dim = x_dim, output_dim = output_dim, num_layers = 1, nh = [64]);

    pi = RNN_Continuous_Policy(x_dim=x_dim, a_dim = a_dim, output_dim=output_dim, nh = 64, LSTM = True, rng=rng);
    #pi = ContinuousPolicy(x_dim = x_dim, output_dim = output_dim, num_layers = 2, nh = [32,32], rng=rng);

    def logger(x,trajs,res):
        #import pdb; pdb.set_trace()
        for k,v in res.items():            
            print k,v

    print 'build updater'
    PiUpdator = VR_Reinforce_PolicyUpdater(pi,max_traj_length, num_trajs, lr = 1e-3, gamma = 0.98, baseline = False);
                                           
    PiUpdator = VR_Reinforce_RNN_PolicyUpdater(policy = pi, 
        max_traj_length = max_traj_length,
        num_trajs = num_trajs, lr = 1e-3, gamma = 0.98, baseline = False);

    PiUpdator = nn_policy_updaters.VRPGPolicyUpdater(pi,
                                                     max_traj_length=max_traj_length, gamma=0.98,
                                                     num_trajs=num_trajs, baseline=nn_policy_updaters.ZeroBaseline(),
                                                     lr=1e-3)                                            
    
    #PiUpdator = nn_policy_updaters.TRPOPolicyUpdater(pi,
    #                                                 max_traj_length=max_traj_length, gamma=0.99,
    #                                                 num_trajs=num_trajs, baseline=nn_policy_updaters.LinearBaseline(),
    #                                                 lr=1e-2)                                            
                                 
          
    learn_policy(PiUpdator, model, env,
                max_traj_len = max_traj_length, num_trajs = num_trajs, 
                num_iter = 1000, logger=logger);
    
