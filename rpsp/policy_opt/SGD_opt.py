import sys

import numpy as np
import theano
import theano.tensor as T

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

optimizers = {'adam': adam, 'RMSProp': RMSProp, 'adadelta': adadelta, 'adagrad': adagrad, 'sgd': sgd}
