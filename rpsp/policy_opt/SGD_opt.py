import numpy as np
from collections import OrderedDict
import cPickle
import theano
import theano.tensor as T
import copy
from IPython import embed
import sys
from rpsp.rpspnets.psr_lite.utils.nn import dbg_print_stats, dbg_print, dbg_nn_assert_notnan, dbg_nn_assert_notinf

sys.setrecursionlimit(1500)


def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX);



def compute_messages(X, kf):
    """
    reshape trajectories
    @param X: X is 3D tensor: Num_of_steps * dx * Num_of_trajs.
    @param kf:
    @return: reshaped values
    """
    X = X[0:kf];
    return np.reshape(X, X.shape[1] * kf);


def Trajs_to_XYpair(Xs, kf):
    """
    Transform trajectories to X, Y values
    @param X: Xs is 3D tensor: Num_of_steps * dx * Num_of_trajs.
    @param kf: max number of steps
    @return:
    """
    tmp_mess = compute_messages(Xs[:, :, 0], kf);
    dim_mess = tmp_mess.shape[0];

    X_tensor = [];
    Y_tensor = [];

    m0 = np.zeros(dim_mess);
    for i in xrange(0, Xs.shape[2]):
        m0 = m0 + compute_messages(Xs[:, :, i], kf);
    m0 = m0 / (Xs.shape[2] * 1.);

    for i in xrange(0, Xs.shape[2]):
        traj = Xs[:, :, i];
        X = traj[0:-kf, :];
        Y = traj[1:traj.shape[0] - kf + 1, :];
        for k in xrange(1, kf):
            Y = np.concatenate((Y, traj[k + 1:traj.shape[0] - kf + k + 1, :]), axis=1);

        X_tensor.append(X);
        Y_tensor.append(Y);

    return np.array(X_tensor), np.array(Y_tensor), m0;  # return 3D tensor: Num_of_trajs * Num of steps * dim


def RMSProp(error, params, learning_rate=1e-2, rho=0.99, epsilon=1e-8, all_grads=[]):
    """
       RMSProp gradient update.
       @param error: loss function
       @param params: list of parameters to update
       @param rho: gradient coefficient in [0,1] if high give more importance to previous vs. current
       @param learning_rate: learning rate of descent direction
       @param epsilon: smoothing coefficient
       @param all_grads: list of pre-computed gradients if len=0 will compute
       @return: list of theano parameter updates (variable, updated_value)
       """
    if len(all_grads) == 0:
        all_grads = T.grad(cost=error, wrt=params);
    updates = [];
    for p, g in zip(params, all_grads):
        # p is param and g is grad
        # construct acc (only once when this function RMSprop is called (only once too)). #every time you construct such acc, theano stores nodes for it.
        acc = theano.shared(p.get_value() * numpy_floatX(0.));
        acc_new = rho * acc + (1 - rho) * (g ** 2);
        gradient_scaling = T.sqrt(acc_new + epsilon);
        g = g / gradient_scaling;
        updates.append((acc, acc_new));
        updates.append((p, p - learning_rate * g));
    return updates;


def sgd(error, params, learning_rate=1e-2, all_grads=[]):
    """
       Classic stochastic gradient update.
       @param error: loss function
       @param params: list of parameters to update
       @param learning_rate: learning rate of descent direction
       @param all_grads: list of pre-computed gradients if len=0 will compute
       @return: list of theano parameter updates (variable, updated_value)
       """
    if len(all_grads) == 0:
        all_grads = T.grad(cost=error, wrt=params);
    updates = [];
    for p, g in zip(params, all_grads):
        updates.append((p, p - learning_rate * g));
    return updates;


def adagrad(error, params, learning_rate=1e-2, epsilon=1e-6, all_grads=[]):
    """
       ADAGRAD gradient update.
       @param error: loss function
       @param params: list of parameters to update
       @param rho: gradient coefficient in [0,1] if high give more importance to previous vs. current
       @param learning_rate: learning rate of descent direction
       @param epsilon: smoothing coefficient
       @param all_grads: list of pre-computed gradients if len=0 will compute
       @return: list of theano parameter updates (variable, updated_value)
       """
    if len(all_grads) == 0:
        all_grads = T.grad(cost=error, wrt=params);
    updates = [];
    for p, g in zip(params, all_grads):
        acc = theano.shared(p.get_value() * numpy_floatX(0.));
        acc_new = acc + g ** 2;
        gradient_scaling = T.sqrt(acc_new + epsilon);
        g = g / gradient_scaling;
        updates.append((acc, acc_new));
        updates.append((p, p - learning_rate * g));
    return updates;


def adadelta(error, params, learning_rate=1e-2, rho=0.9, epsilon=1e-6, all_grads=[]):
    """
       ADADELTA gradient update.
       @param error: loss function
       @param params: list of parameters to update
       @param rho: gradient coefficient in [0,1] if high give more importance to previous vs. current
       @param learning_rate: learning rate of descent direction
       @param epsilon: smoothing coefficient
       @param all_grads: list of pre-computed gradients if len=0 will compute
       @return: list of theano parameter updates (variable, updated_value)
       """
    if len(all_grads) == 0:
        all_grads = T.grad(cost=error, wrt=params);
    updates = [];
    for p, g in zip(params, all_grads):
        acc_g = theano.shared(p.get_value() * numpy_floatX(0.));
        acc_d = theano.shared(p.get_value() * numpy_floatX(0.));
        acc_g_new = rho * acc_g + (1. - rho) * g;
        gradient_scaling = T.sqrt(acc_d + epsilon) / T.sqrt(acc_g_new + epsilon);
        g = g / gradient_scaling;
        acc_d_new = rho * acc_d + (1 - rho) * (g ** 2);
        updates.append((acc_g, acc_g_new));
        updates.append((acc_d, acc_d_new));
        updates.append((p, p - learning_rate * g));
    return updates;


# ADAM:
# def adam_size(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001, epsilon=1e-8, gamma=1-1e-8, clip_bounds=[]):
#     updates = []
#     cost = loss
#     if len(clip_bounds)==2:
#         cost = theano.gradient.grad_clip(loss,clip_bounds[0],clip_bounds[1])
#     all_grads = T.grad(cost = cost, wrt = all_params); 
#     i = theano.shared(np.float32(1))  # HOW to init scalar shared?
#     i_t = i + 1.
#     fix1 = 1. - (1. - beta1)**i_t
#     fix2 = 1. - (1. - beta2)**i_t
#     beta1_t = 1.-(1.- beta1)*gamma**(i_t-1)   # ADDED
#     learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1) 
#     for param_i, g in zip(all_params, all_grads):
#         param_i_value = param_i.get_value()
#         
#         assert not np.isnan(param_i_value).any(), 'param is nan before update adam'
#         m = theano.shared(
#             np.zeros(param_i_value.shape, dtype=theano.config.floatX), name='adam_m_%s' % param_i.name)
#         v = theano.shared( np.zeros(param_i_value.shape, dtype=theano.config.floatX), name='adam_v_%s' % param_i.name)
#         dmin = T.min([param_i.T.shape[0], m.shape.T[0]])
# #         param_iv = T.as_tensor_variable(param_i, name='param_i_%s'%param_i.name)
# #         param_iv = dbg_print_shape('adam_m_%s' % param_iv.name, param_iv)
# #         mv = T.as_tensor_variable(m, name='mv_%s'%m.name)
# #         mv = dbg_print_shape('adam_%s' % mv.name, mv)
# #         vv = T.as_tensor_variable(v, name='vv_%s'%v.name )
# #         vv = dbg_print_shape('adam_%s' % vv.name, vv)
# #         gv = T.as_tensor_variable(g, name='gv_%s'%g.name )
# #         gv = dbg_print_shape('adam_%s' % gv.name, gv)
#         m_t = (beta1_t * g.T[:dmin]) + ((1. - beta1_t) * m.T[:dmin]) # CHANGED from b_t to use beta1_t
#         v_t = (beta2 * g.T[:dmin]**2) + ((1. - beta2) * v.T[:dmin])
#         g_t = m_t / (T.sqrt(T.abs_(v_t)) + epsilon)
#         
#         #pad if necessary dmin <= param.shape[1]
#         g_t_pad =  T.zeros(param_i.shape).T
#         T.set_subtensor(g_t_pad[:dmin],g_t)
#         param_i_t = param_i - (learning_rate_t * g_t_pad.T)
#         m_t_pad =  T.zeros(m.shape).T
#         T.set_subtensor(m_t_pad[:dmin],m_t)
#         v_t_pad =  T.zeros(v.shape).T
#         T.set_subtensor(v_t_pad[:dmin],v_t)
#         
#         updates.append((m, m_t_pad.T))
#         updates.append((v, v_t_pad.T))
#         updates.append((param_i, param_i_t))
#     updates.append((i, i_t))                
#     return updates


# def adam_old(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
#         epsilon=1e-8, gamma=1-1e-8,  clip_bounds=[], all_grads=[]):
#     updates = []
#     if len(all_grads)==0:
#         cost = loss
#         if len(clip_bounds)==2:
#             cost = theano.gradient.grad_clip(loss,clip_bounds[0],clip_bounds[1])
#         all_grads = T.grad(cost = cost, wrt = all_params);
#
#     i = theano.shared(np.float32(1))  # HOW to init scalar shared?
#     i_t = i + 1.
#     fix1 = 1. - (1. - beta1)**i_t
#     fix2 = 1. - (1. - beta2)**i_t
#     beta1_t = 1.-(1.- beta1)*gamma**(i_t-1)   # ADDED
#     learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)
#     for param_i, g in zip(all_params, all_grads):
#         param_i_value = param_i.get_value()
#         #print i, param_i_value
#         assert not np.isnan(param_i_value).any(), 'param is nan before update adam'
#         m = theano.shared(
#             np.zeros(param_i_value.shape, dtype=theano.config.floatX), name='adam_m_%s' % param_i.name)
#         v = theano.shared( np.zeros(param_i_value.shape, dtype=theano.config.floatX), name='adam_v_%s' % param_i.name)
#
#         m_t = ((1. - beta1_t) * g) + (beta1_t * m) # CHANGED from b_t to use beta1_t goes from beta1 to 1.0 over time
#         #m_t = ((beta1) * g) + ((1.-beta1) * m)
#         v_t = (beta2 * g**2) + ((1. - beta2) * v)
#         g_t = m_t / (T.sqrt(T.abs_(v_t)) + epsilon)
#
#         param_i_t = param_i - (learning_rate_t * g_t)
#         updates.append((m, m_t))
#         updates.append((v, v_t))
#         updates.append((param_i, param_i_t))
#     updates.append((i, i_t))
#     return updates


# ADAM:
def adam(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
         epsilon=1e-8, gamma=1 - 1e-8, clip_bounds=[], all_grads=[]):
    """
    ADAM gradient update.
    @param loss: loss function
    @param all_params: list of parameters to update
    @param learning_rate: learning rate of descent direction
    @param beta1: gradient coefficient in [0,1] if high give more importance to current vs. previous
    @param beta2: variance coefficient in [0,1] if high give more importance to current vs. previous
    @param epsilon: smoothing coefficient
    @param gamma: time decay coefficient [0,1]
    @param clip_bounds: gradient clip bounds [low,high]
    @param all_grads: list of pre-computed gradients if len=0 will compute
    @return: list of theano parameter updates (variable, updated_value)
    """
    updates = []
    if len(all_grads) == 0:
        cost = loss
        if len(clip_bounds) == 2:
            cost = theano.gradient.grad_clip(loss, clip_bounds[0], clip_bounds[1])
        all_grads = T.grad(cost=cost, wrt=all_params);

    i = theano.shared(np.float32(1))
    i_t = i + 1.
    fix1 = 1. - (1. - beta1) ** i_t
    fix2 = 1. - (1. - beta2) ** i_t
    # beta1_t = 1.-(1.- beta1)*gamma**(i_t-1)   # ADDED
    learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)
    for param_i, g in zip(all_params, all_grads):
        param_i_value = param_i.get_value()
        assert not np.isnan(param_i_value).any(), 'param is nan before update adam'
        m = theano.shared(
            np.zeros(param_i_value.shape, dtype=theano.config.floatX), name='adam_m_%s' % param_i.name)
        v = theano.shared(np.zeros(param_i_value.shape, dtype=theano.config.floatX), name='adam_v_%s' % param_i.name)
        # m_t = ((1. - beta1_t) * g) + (beta1_t * m) # CHANGED from b_t to use beta1_t goes from 0.1 to 1.0 over time
        m_t = ((beta1) * g) + ((1. - beta1) * m)
        v_t = (beta2 * g ** 2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(T.abs_(v_t)) + epsilon)

        param_i_t = param_i - (learning_rate_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t))
    updates.append((i, i_t))
    return updates


# def natural_sgd(loss, all_params, nabla, curr_params, learning_rate=0.01, epsilon=1e-3):
#     """
#     Natural stochastic gradient update
#     @param loss: loss function
#     @param all_params: list of parameters to update
#     @param nabla: nabla is n x d TODO: fix
#     @param curr_params: current parameters
#     @param learning_rate: gradient learning rate
#     @param epsilon: smoothing inverse parameter
#     @return: list of theano parameter updates (variable, updated_value)
#     """
#     updates = [];
#     all_grads = T.grad(cost=loss, wrt=all_params);
#     flatten_grad_tmp = [p.ravel() for p in all_grads];
#     flatten_grad = T.concatenate(flatten_grad_tmp, axis=0);
#     tmp1 = nabla.dot(flatten_grad);
#     tmp2 = T.inv(epsilon * T.eye(nabla.shape[0]) + nabla.dot(nabla.T)).dot(tmp1);
#     tmp3 = nabla.T.dot(tmp2);
#     new_flatten_params = curr_params - (learning_rate / epsilon) * (flatten_grad - tmp3)
#     t = 0;
#     for param_i, p in zip(all_params, flatten_grad_tmp):
#         param_i_new_flatten = new_flatten_params[t: t + p.shape[0]];
#         t = t + p.shape[0];
#         param_i_new = param_i_new_flatten.reshape(param_i.shape);
#         updates.append((param_i, param_i_new));
#     return updates;
#
#
# def natural_sgd_naive(loss, all_params, nabla, curr_params, KL_Delta=0.1, epsilon=1e-1, M=10):
#     """
#     Natural stochastic gradient update with Fisher estimator
#     @param loss: loss function
#     @param all_params: list of parameters to update
#     @param nabla: nabla is n x d
#     @param curr_params: current parameters
#     @param learning_rate: gradient learning rate
#     @param epsilon: smoothing inverse parameter
#     @param M: number of TODO: fix
#     @param KL_Delta: TODO: fix
#     @return: list of theano parameter updates (variable, updated_value)
#     """
#     updates = [];
#     unbiased_fisher_est = nabla.T.dot(nabla) / M;
#     all_grads = T.grad(cost=loss, wrt=all_params);
#     flatten_grad_tmp = [p.ravel() for p in all_grads];
#     flatten_grad = T.concatenate(flatten_grad_tmp, axis=0);
#     fisher_inverse = T.nlinalg.pinv(unbiased_fisher_est + epsilon * T.eye(nabla.shape[1]));
#     # T.inv(unbiased_fisher_est + epsilon*T.eye(nabla.shape[1]));
#     learning_rate = T.sqrt(KL_Delta / flatten_grad.dot(fisher_inverse).dot(flatten_grad));
#     new_flatten_params = curr_params - learning_rate * fisher_inverse.dot(flatten_grad);
#     t = 0;
#     for param_i, p in zip(all_params, flatten_grad_tmp):
#         param_i_new_flatten = new_flatten_params[t: t + p.shape[0]];
#         t = t + p.shape[0];
#         param_i_new = param_i_new_flatten.reshape(param_i.shape);
#         updates.append((param_i, param_i_new));
#     return updates;

#
# def SGD_train(Xs_train, Xs_test, kf, step_ahead=1, learning_rate=0.1, n_epochs=20, nh=100, method='PSIM',
#               optimizer='sgd',
#               rate_decay_step=5, learning_rate_threshold=1e-5, W_initial=None, rng=None):
#     """
#     Stochastic gradient descent update
#     @param Xs_train:
#     @param Xs_test:
#     @param kf:
#     @param step_ahead:
#     @param learning_rate:
#     @param n_epochs:
#     @param nh:
#     @param method:
#     @param optimizer:
#     @param rate_decay_step:
#     @param learning_rate_threshold:
#     @param W_initial:
#     @param rng:
#     @return:
#     """
#     print 'learning rate {}'.format(learning_rate);
#     Num_train = Xs_train.shape[2];
#     # seperate training data for train+valid.
#     Xtensor_train, Ytensor_train, m_0 = Trajs_to_XYpair(Xs_train[:, :, 0: int(0.9 * Num_train)], kf);
#     Xtensor_valid, Ytensor_valid = Trajs_to_XYpair(Xs_train[:, :, int(0.9 * Num_train):], kf)[0:-1];
#     Xtensor_test, Ytensor_test = Trajs_to_XYpair(Xs_test, kf)[0:-1];
#
#     dx = Xs_train.shape[1];
#     Omegas = rng.multivariate_normal(np.zeros(dx * kf + dx), np.identity(dx * kf + dx), int(1.5 * (dx * kf + dx)));
#     Sigma = numpy_floatX((1. * (dx * kf + dx)) ** 0.5);
#
#     # m_0 = np.zeros(m_0.shape[0]);
#     if method == 'PSIM':
#         if W_initial is None:
#             model = psim_bp.PSIM_BackPropagation(kf, dx, m_0);
#         else:
#             model = psim_bp.PSIM_BackPropagation(kf, dx, m_0, W_x=W_initial[:, 0:dx], W_mess=W_initial[:, dx:],
#                                                  b=np.zeros(m_0.shape[0]));
#
#     elif method == 'RNN':
#         model = psim_bp.RNN(kf, dx, nh, m_0);
#     elif method == 'RNN_PSIM':
#         model = psim_bp.RNN_PSIM(kf, dx, nh, m_0);
#     else:
#         print "Current does not support {}".format(method);
#         assert False;
#
#     # simbolic X and Y, for constructing functions.
#     X = T.matrix();
#     Y = T.matrix();
#     lr = T.scalar('lr');
#
#     symbolic_error = model.go_through_one_traj_return_mse(X, Y);
#     # symbolic_batch_gradient = T.grad(symbolic_error, model.params);
#     # updates = OrderedDict((p, p - lr*g) for p,g in zip(model.params, symbolic_batch_gradient));
#
#     if optimizer == 'sgd':
#         updates = sgd(symbolic_error, model.params, lr);  # using sgd for updates.
#     elif optimizer == 'rmsprop':
#         updates = RMSProp(symbolic_error, model.params, lr);
#     elif optimizer == 'adagrad':
#         updates = adagrad(symbolic_error, model.params, lr);
#     elif optimizer == 'adadelta':
#         updates = adadelta(symbolic_error, model.params, lr);
#     else:
#         print "no such optimizer is available now";
#         assert False;
#
#     train_model = theano.function(inputs=[X, Y, lr], outputs=symbolic_error, updates=updates);
#
#     sym_step = T.lscalar();
#     sym_messs, sym_error = model.go_through_one_traj_return_messs(X, Y, sym_step);
#     forward_prediction = theano.function(inputs=[X, Y, sym_step], outputs=[sym_messs, sym_error]);
#
#     # now ready to do SGD:
#     print '... ... training the model with gradient descent'
#     be = -1;  # keeping track the best epoch so far.
#     min_error_valid = np.inf;
#     best_model = copy.deepcopy(model);  # keeping track the best model.
#     test_error_from_best_model = np.inf;
#     initial_error = 0.;
#     for e in range(0, n_epochs):
#
#         # compute TEST error:
#         test_error = 0;
#         for traj_i in xrange(0, Xtensor_test.shape[0]):
#             test_error = test_error + forward_prediction(Xtensor_test[traj_i], Ytensor_test[traj_i], step_ahead)[1];
#         test_error = test_error / Xtensor_test.shape[0];
#         print '[Testing] test error {} at epoch {}'.format(test_error, e);
#         if e == 0:
#             initial_error = test_error;
#
#         # each traj is consider to be a mini-batch.
#         for traj_i in xrange(0, Xtensor_train.shape[0]):
#             train_model(Xtensor_train[traj_i], Ytensor_train[traj_i], learning_rate);
#         # print '[Learning] epoch {}, trajectory {} among total trajectories {}\r'.format(e, traj_i, Xtensor_train.shape[0]);
#         # sys.stdout.flush();
#
#         # compute VALIDATION error:
#         valid_error = 0.;
#         for traj_i in xrange(0, Xtensor_valid.shape[0]):
#             valid_error = valid_error + forward_prediction(Xtensor_valid[traj_i], Ytensor_valid[traj_i], step_ahead)[1];
#         valid_error = valid_error / Xtensor_valid.shape[0];
#         print '[Validation] validation error {} at epoch {}'.format(valid_error, e);
#
#         # check if we find improvement on the validation set.
#         if valid_error < min_error_valid:  # if imporvement on validation set is found:
#             print "validation improvement found in epoch {}, from {} to {}".format(e, min_error_valid, valid_error);
#             best_model = copy.deepcopy(model);  # update the best model found so far.
#             test_error_from_best_model = test_error;  # re
#             min_error_valid = valid_error;  # update the minimum validation error corresponding to the current best model.
#             be = e;  # update the epoch that corresponds to the current best model.
#
#         else:  # no validation improvement found in this epoch:
#             print "No improvement found on validation set in this epoch...";
#
#         if abs(
#                 be - e) > rate_decay_step:  # if no improvement found among the last 5 epoch, we decay the learning rate half.
#             print "No improvement found on the validation set in the last {} epoch, decay learning rate from {} to {}".format(
#                 rate_decay_step,
#                 learning_rate, learning_rate * 0.5);
#             learning_rate = learning_rate * 0.5;
#             # model = copy.deepcopy(best_model); #restore to the recorded best model, with the new learning rate.
#             # model = best_model;
#             model.model_copy(best_model);
#
#         if learning_rate <= learning_rate_threshold:
#             print "early stopping happen: learning rate is less than {}".format(learning_rate_threshold);
#             break;
#
#     model = best_model;  # simply link to the best model, wihtout deepcopying.
#     print "test error from the best model measured on validation set is {}".format(test_error_from_best_model);
#     # save model:
#     filename = 'results/' + method + 'kf{}_step{}_nh{}.p'.format(kf, step_ahead, nh);
#     # cPickle.dump([model.params, test_error_from_best_model], open(filename, 'wb'));
#     return model, test_error_from_best_model, initial_error;
#
#
# def simple_SGD_train(Xs_train, Xs_test, kf, step_ahead=1, learning_rate=0.1, n_epochs=20, nh=100, method='PSIM',
#                      optimizer='sgd',
#                      rate_decay_step=5, learning_rate_threshold=1e-5, W_initial=None, rng=None):
#     print 'learning rate {}'.format(learning_rate);
#     Num_train = Xs_train.shape[2];
#     # seperate training data for train+valid.
#     Xtensor_train, Ytensor_train, m_0 = Trajs_to_XYpair(Xs_train[:, :, 0: int(0.9 * Num_train)], kf);
#     Xtensor_valid, Ytensor_valid = Trajs_to_XYpair(Xs_train[:, :, int(0.9 * Num_train):], kf)[0:-1];
#     Xtensor_test, Ytensor_test = Trajs_to_XYpair(Xs_test, kf)[0:-1];
#
#     dx = Xs_train.shape[1];
#
#     Omegas = rng.multivariate_normal(np.zeros(dx * kf + dx), np.identity(dx * kf + dx), int(1.5 * (dx * kf + dx)));
#     Sigma = np.float32((1. * (dx * kf + dx)) ** 0.5);
#
#     # m_0 = np.zeros(m_0.shape[0]);
#     if method == 'PSIM':
#         if W_initial is None:
#             model = psim_bp.PSIM_BackPropagation(kf, dx, m_0);
#         else:
#             model = psim_bp.PSIM_BackPropagation(kf, dx, m_0, W_x=W_initial[:, 0:dx], W_mess=W_initial[:, dx:],
#                                                  b=np.zeros(m_0.shape[0]));
#
#     elif method == 'RNN':
#         model = psim_bp.RNN(kf, dx, nh, m_0);
#     elif method == 'RNN_PSIM':
#         model = psim_bp.RNN_PSIM(kf, dx, nh, m_0);
#     else:
#         print "Current does not support {}".format(method);
#         assert False;
#
#     # simbolic X and Y, for constructing functions.
#     X = T.matrix();
#     Y = T.matrix();
#     lr = T.scalar('lr');
#
#     symbolic_error = model.go_through_one_traj_return_mse(X, Y);
#     # symbolic_batch_gradient = T.grad(symbolic_error, model.params);
#     # updates = OrderedDict((p, p - lr*g) for p,g in zip(model.params, symbolic_batch_gradient));
#
#     if optimizer == 'sgd':
#         updates = sgd(symbolic_error, model.params, lr);  # using sgd for updates.
#     elif optimizer == 'rmsprop':
#         updates = RMSProp(symbolic_error, model.params, lr);
#     elif optimizer == 'adagrad':
#         updates = adagrad(symbolic_error, model.params, lr);
#     elif optimizer == 'adadelta':
#         updates = adadelta(symbolic_error, model.params, lr);
#     else:
#         print "no such optimizer is available now";
#         assert False;
#
#     train_model = theano.function(inputs=[X, Y, lr], outputs=symbolic_error, updates=updates);
#
#     sym_step = T.lscalar();
#     sym_messs, sym_error = model.go_through_one_traj_return_messs(X, Y, sym_step);
#     forward_prediction = theano.function(inputs=[X, Y, sym_step], outputs=[sym_messs, sym_error]);
#
#     # now ready to do SGD:
#     print '... ... training the model with gradient descent'
#     test_error = np.inf;
#     initial_error = 0.;
#     for e in range(0, n_epochs):
#
#         # compute TEST error:
#         test_error = 0;
#         for traj_i in xrange(0, Xtensor_test.shape[0]):
#             test_error = test_error + forward_prediction(Xtensor_test[traj_i], Ytensor_test[traj_i], step_ahead)[1];
#         test_error = test_error / Xtensor_test.shape[0];
#         print '[Testing] test error {} at epoch {}'.format(test_error, e);
#         if e == 0:
#             initial_error = test_error;
#
#         # each traj is consider to be a mini-batch.
#         for traj_i in xrange(0, Xtensor_train.shape[0]):
#             train_model(Xtensor_train[traj_i], Ytensor_train[traj_i], learning_rate);
#         # print '[Learning] epoch {}, trajectory {} among total trajectories {}\r'.format(e, traj_i, Xtensor_train.shape[0]);
#         # sys.stdout.flush();
#
#         if e % 100 == 0:
#             learning_rate = learning_rate * 0.5;
#
#     print "test error is {}".format(test_error);
#     # save model:
#     filename = 'results/' + method + 'kf{}_step{}_nh{}.p'.format(kf, step_ahead, nh);
#     # cPickle.dump([model.params, test_error_from_best_model], open(filename, 'wb'));
#     return model, test_error, initial_error;
#
#
# def SGD_test(model, Xs_test, kf, step_ahead=1):
#     sym_X = T.matrix();
#     sym_Y = T.matrix();
#     sym_step = T.lscalar();
#     sym_messs, sym_error = model.go_through_one_traj_return_messs(sym_X, sym_Y, sym_step);
#     forward_prediction = theano.function(inputs=[sym_X, sym_Y, sym_step], outputs=[sym_messs, sym_error]);
#
#     Xtensor_test, Ytensor_test = Trajs_to_XYpair(Xs_test, kf)[0:-1];
#     test_errors = [];
#     predicted_trajs = [];
#     for traj_i in xrange(0, Xtensor_test.shape[2]):
#         pred_traj, error = forward_prediction(Xtensor_train[traj_i], Xtensor_test[traj_i], step_ahead);
#         predicted_trajs.append(pred_traj);
#         test_errors.append(error);
#
#     print "test error is {}".format(np.mean(test_error));
#     Pt = np.array(predicted_trajs);
#     return np.mean(test_error), Pt;


optimizers = {'adam': adam, 'RMSProp': RMSProp, 'adadelta': adadelta, 'adagrad': adagrad, 'sgd': sgd}
