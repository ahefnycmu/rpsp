from __future__ import print_function
from p3 import *

import numpy as np
import theano
import theano.tensor as T
import theano.printing
import numpy.linalg as npla
from rpsp import globalconfig


def reshape_mat_f(x, shape):
    '''
    Reshape a symbolc matrix in Fortran order (fill columns first).
    '''
    return T.transpose(T.reshape(T.transpose(x), (shape[1], shape[0])))


def row_kr_product(X, Y, name=None):
    '''
    Row Khatri-row product (Row-wise kronecker product) of two matrices.
    '''
    if X.ndim == 1 and Y.ndim == 1:
        output = T.outer(X, Y).reshape([-1])
    else:
        assert X.ndim == 2 and Y.ndim == 2
        output = (X.reshape((X.shape[0], X.shape[1], 1)) * Y.reshape((Y.shape[0], 1, Y.shape[1]))).reshape(
            (X.shape[0], X.shape[1] * Y.shape[1]))

    if name is not None:
        output.name = name

    return output


def test_theano_func(i, node, fn):
    for output in fn.outputs:
        theano.printing.debugprint(node)
        print('Inputs : %s' % [input[0] for input in fn.inputs])
        print('Outputs: %s' % [output[0] for output in fn.outputs])


class BatchedMatrixInverse(theano.Op):
    '''
    A Batched adaptation of Theano's MatrixInverse:
    Given a 3-mode tensor A, compute a tensor B such that B[i,:,:] = inv(A[i,:,:])
    '''
    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        assert x.ndim == 3
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = npla.inv(x).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
            .. math:: V\frac{\partial X^{-1}}{\partial X},
        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to
            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.
        """
        x, = inputs
        xi = self(x)
        gz, = g_outputs
        # TT.dot(gz.T,xi)

        a = T.batched_dot(xi, gz.transpose(0, 2, 1))
        b = T.batched_dot(a, xi)
        return [-b.transpose(0, 2, 1)]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return
            .. math:: \frac{\partial X^{-1}}{\partial X}V,
        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to
            .. math:: X^{-1} \cdot V \cdot X^{-1}.
        """
        x, = inputs
        xi = self(x)
        ev, = eval_points
        if ev is None:
            return [None]

        a = T.batched_dot(xi, ev)
        b = T.batched_dot(a, xi)
        return [-b]

    def infer_shape(self, node, shapes):
        return shapes


batched_matrix_inverse = BatchedMatrixInverse()


class TheanoOp(theano.Op):
    __props__ = ()

    def __init__(self, callback):
        self._callback = callback
        self.view_map = {0: [0]}

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)

        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        # self.x = x
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = self._callback(x)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return output_grads

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class CallbackOp(theano.Op):
    __props__ = ()

    def __init__(self, callback):
        self._callback = callback
        self.view_map = {0: [0]}

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)

        # Note: using x_.type() is dangerous, as it copies x's broadcasting
        # behaviour
        # self.x = x
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x
        self._callback(x)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return output_grads

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


def dbg_print_shape(msg, X):
    '''
    Given a symbolic variable X, return an identical variable.
    Using the returned variable instead of X prints the shape of X as a side-effect.
    
    Typical usage:
        x = T.vector()
        x = dbg_print_shape("x.shape=", x)        
    '''

    def fn(x):
        print(msg, x.shape, sep='')

    return CallbackOp(fn)(X)


def dbg_check_error(msg, X):
    def fn(x):
        R = np.sum(np.abs(np.sum(x, axis=1)) < 1e-6)
        print(msg, R, sep='')

    return CallbackOp(fn)(X)


def dbg_print(msg, X):
    def fn(x):
        print(msg, x, x.shape, sep='')

    return CallbackOp(fn)(X)


def dbg_print_stats(msg, X):
    def fn(x):
        print(msg, 'min:', np.min(x), 'max:', np.max(x), \
              'avg:', np.mean(x), 'std:', np.std(x), 'shape:', x.shape, sep=' ')

    return CallbackOp(fn)(X)


def dbg_nn_assert(X, condition, msg):
    def fn(x):
        assert condition(x), msg

    return CallbackOp(fn)(X)


def dbg_nn_assert_notnan(X, msg=None):
    if msg is None:
        msg = '%s is None' % X.name
    return dbg_nn_assert(X, lambda x: not np.isnan(np.sum(x)), msg)


def dbg_nn_assert_notinf(X, msg):
    return dbg_nn_assert(X, lambda x: not np.isinf(np.sum(x)), msg)


def tf_get_normalized_grad(loss, params, beta=0.9, normalize=True, clip_bounds=[]):
    '''
    Given a loss function 'loss', computes the gradient normalized
    so that its expected square sum is equal to 1. The expected square sum
    is computed using exponential averaging with parameter 'beta'.
    
    The function returns a topic consisting of:
        1. A symbolic variable list representing the normalized gradient
        2. The normalization weight (multiplicative)
        3. A list of updates to maintain normalization weight        
        
    Normalization can be disabled by passing normalize=False.
    '''
    if len(clip_bounds) == 2 and clip_bounds[1] <> 0.0:
        loss = theano.gradient.grad_clip(loss, clip_bounds[0], clip_bounds[1])
    grad = T.grad(cost=loss, wrt=params, disconnected_inputs='ignore')

    if normalize:
        loss_name = loss.name
        if loss_name is None: loss_name = 'loss'
        var = theano.shared(1.0, name='%s_g2' % loss_name)
        grad_sq = T.sum([T.sum(g ** 2) for g in grad])
        var_new = beta * var + (1.0 - beta) * grad_sq
        weight = 1.0 / T.sqrt(var_new)
        normalized_grad = [gg * weight for gg in grad]
        updates = [(var, var_new)]
    else:
        weight = T.ones([])
        normalized_grad = grad
        updates = []

    return normalized_grad, weight, updates


def tf_get_normalized_grad_per_param(loss, params, beta=0.1, normalize=True, clip_bounds=[]):
    '''
    Given a loss function 'loss', computes the gradient normalized
    so that its expected square sum is equal to 1. The expected square sum
    is computed using exponential averaging with parameter 'beta' for each parameter individually.
    
    The function returns a topic consisting of:
        1. A symbolic variable list representing the normalized gradient
        2. The normalization weight (multiplicative)
        3. A list of updates to maintain normalization weight        
        
    Normalization can be disabled by passing normalize=False.
    '''
    loss_name = loss.name
    if loss_name is None: loss_name = 'loss'
    g_clip = 1.0
    if len(clip_bounds) == 2 and clip_bounds[1] <> 0.0:
        loss = theano.gradient.grad_clip(loss, clip_bounds[0], clip_bounds[1])
        g_clip = clip_bounds[1]
    # grad = T.grad(cost=loss, wrt=params, disconnected_inputs='ignore' )
    updates = [];
    vars_new = T.ones([]);
    grad = [];
    powerg = globalconfig.vars.args.powerg  # 2
    if normalize:
        for param in params:
            cost = loss
            var = theano.shared(1.0, name='%s_g2' % param.name)
            g = T.grad(cost=cost, wrt=param, disconnected_inputs='ignore')
            grad.append(g)
            gsq = T.abs_(g) ** 2
            threshold_var = globalconfig.vars.args.threshold_var
            if threshold_var > 0.0 and threshold_var < 1.0:
                gsq = t_clip_percentile('g2_' + param.name, gsq, threshold_var)
            g2 = T.sum(gsq)
            var_new = beta * var + (1. - beta) * g2
            updates.extend([(var, var_new)])
            vars_new = vars_new + var_new
        weight = 1.0 / (vars_new ** (1.0 / powerg))
        var_clip = globalconfig.vars.args.var_clip
        if var_clip > 0.0:
            weight = T.clip(weight, 1.0 / var_clip, var_clip)
        normalized_grad = [gg * weight for gg in grad]
    else:
        weight = T.ones([])
        normalized_grad = T.grad(cost=loss, wrt=params, disconnected_inputs='ignore')
        updates = []
        if squashg:
            normalized_grad = [T.tanh(gg) * g_clip for gg in normalized_grad]

    return normalized_grad, weight, updates


def t_uncentered_var(param, beta, grad=None):
    weight = T.sum(grad ** 2)
    weight_t = beta * var + (1. - beta) * weight  # beta_t initially is practically 0 and decays if we use beta_t
    updates = [(var, weight_t)]
    return updates, weight_t


def t_clip_percentile(msg, X, threshold=0.9):
    def fn(x):
        if threshold == 1.0:
            return x
        pval = np.percentile(x, threshold)
        # print(msg + '#>{}= {}/{}, sum>{}={}, sum<{}={}'.format(threshold, np.sum(x > pval), np.prod(x.shape), threshold,
        #                                                        np.sum(x * (x > pval)), threshold,
        #                                                        np.sum(x * (x <= pval))))
        x = x * (x < pval)
        return x

    return TheanoOp(fn)(X)


def t_sumsq_grad(loss, param, beta=0.1, clip_bounds=[], grad=None):
    '''
    Computes an exp. averaging of the uncentered variance of param gradient.
    Returns updates, variances and gradient
    '''
    var = theano.shared(1.0, name='%s_g2' % param.name)
    if grad is None:
        if len(clip_bounds) == 2 and clip_bounds[1] <> 0.0:
            loss = theano.gradient.grad_clip(loss, clip_bounds[0], clip_bounds[1])
        grad = T.grad(cost=loss, wrt=param, disconnected_inputs='ignore')
        # grad = dbg_nn_assert_notnan(grad,'grad_%s is NAN!!'%param.name)
    g2 = grad ** 2
    # g2 = dbg_print_stats('g2', g2)
    threshold_var = globalconfig.vars.args.threshold_var
    if threshold_var > 0.0 and threshold_var < 1.0:
        g2 = t_clip_percentile('g2_' + param.name, g2, threshold_var)
    # g2 = dbg_print_stats('g2after', g2)
    weight = T.sum(g2)
    weight_t = beta * var + (1. - beta) * weight  # beta_t initially is practically 0 and decays if we use beta_t
    updates = [(var, weight_t)]
    return updates, weight_t, grad


def tf_grad_updates(loss1, loss2, params, beta=0.1, clip_bounds=[]):
    updates = [];
    grads = [];
    weights = [];
    g1 = 0.;
    g2 = 0.;
    w2 = 0.;
    w1 = 0.;
    for param_i in params:
        if loss1 <> 0.0:
            updates1, w1, g1 = t_sumsq_grad(loss1, param_i, beta=beta, clip_bounds=clip_bounds)
            updates.extend(updates1)
        if loss2 <> 0.0:
            updates2, w2, g2 = t_sumsq_grad(loss2, param_i, beta=beta, clip_bounds=clip_bounds)
            updates.extend(updates2)
        weights.append((w1, w2))
        grads.append((g1, g2))
    return updates, grads, weights


#
# def get_grad_update_old( loss1, loss2, params, c1=1., c2=1., epsilon_min = 1e-7, epsilon_max=1e7, beta=0.1, clip_bounds=[]):
#     alpha_1 = theano.shared(np.float64(1))
#     alpha_2 = theano.shared(np.float64(1))
#     combined_grads= [];
#     updates,grads,weights = tf_grad_updates(loss1, loss2, params, beta=beta, clip_bounds=clip_bounds)
#
#     #alpha_l1_t = (T.sum(zip(*weights)[0])).clip(epsilon_min,epsilon_max)
#     #alpha_l2_t = (T.sum(zip(*weights)[1])).clip(epsilon_min,epsilon_max)
#     alpha_l1_t = T.sum(zip(*weights)[0])
#     alpha_l2_t = T.sum(zip(*weights)[1])
#     #alpha_l2_t = dbg_nn_skip_update(alpha_l2_t, lambda x: x>epsilon_max, 'w2 too large skip update')
#     w1 = T.clip(1/T.sqrt(alpha_l1_t ),epsilon_min,epsilon_max)  if loss1<>0.0 else 0.0
#     w2 = T.clip(1/T.sqrt(alpha_l2_t ),epsilon_min,epsilon_max)  if loss2<>0.0 else 0.0
#
#     for (g1,g2) in grads:
#         combined_grad = g1*w1*c1 + g2*w2*c2
#         combined_grads.append(combined_grad)
#     combined_loss = loss1*w1*c1 + loss2*w2*c2
#     updates.extend([(alpha_1, alpha_l1_t)])
#     updates.extend([(alpha_2, alpha_l2_t)])
#     #combined_loss = dbg_print('combined_loss', combined_loss)
#     #combined_loss = dbg_nn_assert(combined_loss, lambda x: x<1e5, 'too large loss')
#     results={'total_cost':combined_loss, 'cost2_avg':loss2*w2*c2 ,\
#             'cost1_avg':loss1*w1*c1, 'a1':alpha_l1_t,'a2':alpha_l2_t,\
#             'updates':updates, 'grads':combined_grads,'params':params, 'total_grads':grads}
#     return results

def cg_solve(A, b, iter=10, reg=0.0, use_scan=False):
    x0 = T.zeros_like(b)
    r0 = b

    if use_scan:
        result, updates = theano.scan(fn=_cg_step, outputs_info=[x0, r0, r0],
                                      non_sequences=[A, reg], n_steps=iter, strict=True)
        return result[0][-1]
    else:
        # Use for loop (slower compilation but faster execution)
        xi = x0
        ri = r0
        pi = r0

        for i in xrange(iter):
            xi, ri, pi = _cg_step(xi, ri, pi, A, reg)

        return xi


def _cg_step(x, r, p, A, reg):
    Ap = T.dot(A, p) + p * reg
    r2old = T.dot(r.T, r)
    alpha_k = r2old / T.dot(p.T, Ap)
    x_k = x + alpha_k * p
    r_k = r - alpha_k * Ap
    r2new = T.dot(r_k.T, r_k)
    p_k = r_k + (r2new / r2old) * p
    return x_k, r_k, p_k


def cg_solve_batch(AA, B, iter=10, reg=0.0, use_scan=False):
    '''
    Solves a batch of conjugate gradient problems.
    AA is a tensor and B is amatrix such as AA[i] and B[i]
    specify the ith problem.
    '''
    X0 = T.zeros_like(B)
    R0 = B

    if use_scan:
        result, _ = theano.scan(fn=_cg_step_batch, outputs_info=[X0, R0, R0], non_sequences=[AA, reg], n_steps=iter,
                                strict=True)
        return result[0][-1]
    else:
        Xi = X0
        Ri = R0
        Pi = R0

        for i in xrange(iter):
            Xi, Ri, Pi = _cg_step_batch(Xi, Ri, Pi, AA, reg)

        return Xi


def _cg_step_batch(X, R, P, AA, reg):
    AP = T.batched_dot(AA, P) + P * reg
    r2old = T.sum(R * R, axis=1)
    pAp = T.sum(P * AP, axis=1)
    alpha_k = T.reshape(r2old / pAp, (-1, 1))
    X_k = X + alpha_k * P
    R_k = R - alpha_k * AP
    r2new = T.sum(R_k * R_k, axis=1)
    P_k = R_k + T.reshape(r2new / r2old, (-1, 1)) * P
    return X_k, R_k, P_k


def mia(A, it=10, F=None, reg=0.0):  # numpy version for rffpsr
    d = A.shape[0]
    if F is None: F = np.sqrt(np.sum(A * A))

    F = F + reg
    G = A / F

    Y = np.eye(d) * (1 - reg / F) - G
    Z = [None] * it
    Z[0] = np.eye(d)
    for i in xrange(1, it):
        Z[i] = np.dot(Z[i - 1], Y)

    output = sum(Z) / F
    return output


def neumann_inv(A, it=10, F=None, reg=0.0):  # theano version for rffpsr_rnn
    d = A.shape[0]
    if F is None: F = T.sqrt(T.sum(A * A))

    F = F + reg
    G = A / F

    Y = T.eye(d) * (1 - reg / F) - G
    Z = [None] * it
    Z[0] = T.eye(d)
    for i in xrange(1, it):
        Z[i] = T.dot(Z[i - 1], Y)

    output = sum(Z) / F
    return output


def neumann_inv_batch(A, it=10, F=None, reg=0.0):
    N, d, _ = A.shape
    if F is None:
        F = T.sqrt(T.sum(A * A, axis=(1, 2)))
        F = T.reshape(F, (N, 1, 1))
        F = T.tile(F, (1, d, d))

    G = A / F
    Y = T.tile(T.eye(d), (N, 1, 1)) * (1 - reg / F) - G
    Z = [None] * it
    Z[0] = T.tile(T.eye(d), (N, 1, 1))
    for i in xrange(1, it):
        Z[i] = T.batched_dot(Z[i - 1], Y)

    output = sum(Z) / F
    return output


class TheanoOptimizer:
    def init(self, output_tf, inputs, params, on_unused_input='raise'):
        raise NotImplementedError

    def step(self, inputs, step_szie=1.0):
        raise NotImplementedError

    def get_context(self):
        raise NotImplementedError

    def reset_context(self, context):
        raise NotImplementedError


class TheanoSGDOptimizer(TheanoOptimizer):
    def __init__(self, eta):
        self._eta = eta

    def init(self, output_tf, inputs, params, on_unused_input='raise'):
        step_size = T.scalar()
        output_tf = T.sum([output_tf])
        grads = [T.grad(output_tf, p) for p in params]
        updates = [(pg[0], pg[0] - step_size * self._eta * pg[1]) for pg in zip(params, grads)]
        self._fn = theano.function(inputs=inputs + [step_size], outputs=output_tf, updates=updates,
                                   on_unused_input=on_unused_input)

    def step(self, inputs, step_size=1.0):
        return self._fn(*(inputs) + [step_size])

    def get_context(self):
        return None

    def reset_context(self, context):
        pass


def eval_CG(iter=10):
    A = T.matrix('A')
    b = T.vector('b')
    # C = T.matrix('C')
    # fn = theano.function(inputs=[A,b], outputs=CGD_optimizer(A,b, iter), allow_input_downcast=True)
    fn = theano.function(inputs=[A, b], outputs=cg_solve(A, b, iter), allow_input_downcast=True)
    return fn


def CGD_num_optimizer(As, bs, iter=10, C=None):
    symmetrize = True
    if As.shape[0] <> As.shape[1]:
        symmetrize = True
    elif not np.allclose(As.T, As):
        symmetrize = True
    if symmetrize:
        print('symetrize num')
        A = np.copy(np.dot(As.T, As))
        b = np.copy(np.dot(As.T, bs))

    C = np.eye(b.shape[0]) if C is None else C
    x = np.zeros(b.shape[0])
    r = b - np.dot(A, x)
    p = b - np.dot(A, x)
    for i in xrange(iter):
        z = np.dot(r, C)
        Ap = np.dot(A, p)
        r2old = np.dot(r.T, z)
        alpha_k = r2old / np.dot(p.T, Ap)
        x = x + alpha_k * p
        r = r - alpha_k * Ap
        z = np.dot(r, C)
        r2new = np.dot(r.T, z)
        p = r + (r2new / r2old) * p
    return x


# faster preconditioner not tested
def CGD_optimizer(As, bs, iter=10, C=None):
    symmetrize = False
    if As.shape[0] <> As.shape[1]:
        symmetrize = True
    elif not T.allclose(As.T, As):
        symmetrize = True
    if symmetrize:
        A = T.dot(As.T, As)
        b = T.dot(As.T, bs)
    C = T.eye(b.shape[0]) if C is None else C
    x0 = T.zeros_like(b)
    r0 = b - T.dot(A, x0)
    z0 = T.dot(C, r0)
    result, updates = theano.scan(fn=CG_single_step, outputs_info=[x0, r0, z0], non_sequences=[A, C], n_steps=iter,
                                  strict=True)
    return result[0][-1]


def CG_single_step(x, r, p, A, C):
    z = T.dot(r, C)
    Ap = T.dot(A, p)
    r2old = T.dot(r.T, z)
    alpha_k = r2old / T.dot(p.T, Ap)
    x_k = x + alpha_k * p
    r_k = r - alpha_k * Ap
    z_k = T.dot(r_k, C)
    r2new = T.dot(r_k.T, z_k)
    p_k = z_k + (r2new / r2old) * p
    return x_k, r_k, p_k


''' 
Solve linear system using conjugate gradient. Early stop in iter iterations
Ax=b if A non symmetric pass : A:=A^TA b:=A^Tb
A,b are np arrays
C is preconditioner. pass C^-1 inverse if left preconditioner. C=None if no preconditioning
'''


class TheanoCGDOptimizer(TheanoOptimizer):  # too slow
    def __init__(self, loss=None, inputs=None, C=None):
        symmetrize = False
        A, b = inputs
        if A.shape[0] <> A.shape[1]:
            symmetrize = True
        elif not T.allclose(A.T, A):
            print('not sym th')
            symmetrize = True
        if symmetrize:
            print('symetrize thean')
            self._A = T.dot(A.T, A)
            self._b = T.dot(A.T, b)
        else:
            self._A = A
            self._b = b
        #         self._A = theano.shared(A)
        #         self._b = theano.shared(b)
        if C is None:
            self._C = T.eye(self._A.shape[1])
        else:
            self._C = C
        b = self._b.eval()
        A = self._A.eval()
        self._x0 = np.zeros(b.shape[0])
        self._r0 = b - np.dot(A, self._x0)
        # self._z = T.dot(self._C,theano.shared(self._x0))
        self._t_x = theano.shared(self._x0)  # T.vector('x')
        self._output_tf = loss
        if loss is None:
            self._output_tf = self._tf_CG_loss()

    def _tf_CG_loss(self):
        return 0.5 * T.dot(T.dot(self._t_x.T, self._A), self._t_x) - T.dot(self._t_x.T, self._b)

    def init(self, output_tf, inputs, params, on_unused_input='raise'):
        alpha = theano.shared(np.float64(1.0));
        r = theano.shared(self._r0);
        p = theano.shared(self._r0);
        updates = [];
        ##conjugate gradient update
        z = T.dot(r, self._C)
        Ap = T.dot(self._A, p)
        r2old = T.dot(r.T, z)
        alpha_k = r2old / T.dot(p.T, Ap)
        x_k = self._t_x + alpha_k * p
        r_k = r - alpha_k * Ap
        z_k = T.dot(r_k, self._C)
        r2new = T.dot(r_k.T, z_k)
        p_k = z_k + (r2new / r2old) * p
        updates.append((alpha, alpha_k))
        updates.append((r, r_k));
        updates.append((p, p_k));
        updates.append((self._t_x, x_k));
        self._fn = theano.function(inputs=[], outputs=output_tf, updates=updates, on_unused_input=on_unused_input)
        return

    def step(self, inputs, step_size=1.0):
        return self._fn()

    def get_context(self):
        return None

    def reset_context(self, context):
        pass

    def minimize(self, iter=10):
        sampler = lambda x: []
        minimize_theano_fn(self._output_tf, [], (),
                           1, sampler, max_iterations=iter, on_unused_input='raise', optimizer=self)
        return self._t_x


class TheanoAdamOptimizer:
    def __init__(self, b1=0.9, b2=0.999, alpha=0.001):
        self._b1 = b1
        self._b2 = b2
        self._b1t = theano.shared(b1)
        self._b2t = theano.shared(b2)
        self._t = theano.shared(0)
        self._alpha = alpha

    def init(self, output_tf, inputs, params, on_unused_input='raise'):
        self._mean = [theano.shared(np.zeros(p.get_value().shape)) for p in params]
        self._var = [theano.shared(np.zeros(p.get_value().shape)) for p in params]

        step_size = T.dscalar()
        grads = [T.grad(output_tf, p) for p in params]

        new_mean = [mg[0] * self._b1 + (1 - self._b1) * mg[1] for mg in zip(self._mean, grads)]
        new_var = [vg[0] * self._b2 + (1 - self._b2) * vg[1] * vg[1] for vg in zip(self._var, grads)]

        mh = [m / (1 - self._b1t) for m in new_mean]
        vh = [v / (1 - self._b2t) for v in new_var]

        diff = [mv[0] / (T.sqrt(mv[1]) + 1e-8) for mv in zip(mh, vh)]
        updates = [(pd[0], pd[0] - pd[1] * step_size * self._alpha) for pd in zip(params, diff)]
        updates += list(zip(self._mean, new_mean)) + list(zip(self._var, new_var))
        updates += [(self._b1t, self._b1t * self._b1), (self._b2t, self._b2t * self._b2)]
        self._fn = theano.function(inputs=inputs + [step_size], outputs=output_tf, updates=updates,
                                   on_unused_input=on_unused_input)

    def step(self, inputs, step_size=1.0):
        return self._fn(*(inputs) + [step_size])

    def get_context(self):
        return [self._b1t.get_value(), self._b2t.get_value(), self._t.get_value()] + \
               [m.get_value() for m in self._mean] + [v.get_value() for v in self._var]

    def reset_context(self, context):
        self._b1t.set_value(context[0])
        self._b2t.set_value(context[1])
        self._t.set_value(context[2])

        idx = 3
        for m in self._mean:
            m.set_value(context[idx])
            idx += 1

        for v in self._var:
            v.set_value(context[idx])
            idx += 1


def minimize_theano_fn(output_tf, input_vars, params,
                       num_samples, sampler, validation_batch=5,
                       num_val_samples=0, val_sampler=None,
                       initial_step=0.1,
                       min_step=1e-5, max_iterations=100, logger=None,
                       on_unused_input='raise', optimizer='sgd'):
    def copy_params(src, dst):
        for j in xrange(len(src)):
            dst[j].set_value(src[j].get_value())

    if optimizer == 'sgd':
        optimizer = TheanoSGDOptimizer(1.0)
    elif optimizer == 'adam':
        optimizer = TheanoAdamOptimizer()
    else:
        assert isinstance(optimizer, TheanoOptimizer)

    optimizer.init(output_tf, input_vars, params, on_unused_input=on_unused_input)

    eval_fn = theano.function(inputs=input_vars, outputs=output_tf, on_unused_input=on_unused_input)

    if max_iterations < 0:
        max_iterations = np.inf
    if min_step > initial_step:
        min_step = initial_step

    it = 0

    if num_val_samples > 0:
        best_it = 0
        # last_batch_val_obj = np.inf
        best_params = [theano.shared(p.get_value()) for p in params]
        context = optimizer.get_context()

        print('Initial Validation Objective = ', end='')
        init_val = sum(eval_fn(*val_sampler(i)) for i in xrange(num_val_samples))
        best_val = init_val
        batch_val_obj = np.inf
        last_batch_val_obj = best_val
        print(init_val)
    print('max_iterations, ', max_iterations)
    step = initial_step
    while it < max_iterations:
        #print('iter minimize_theano_fn ', it)
        for i in xrange(num_samples):
            inputs = sampler(i)
            optimizer.step(inputs, step)
        train_obj = np.sum([eval_fn(*sampler(i)) for i in xrange(num_samples)])
        print('minimize_theano_fn::Iteration %d - train Objective = %e' % (it, train_obj))
        val_obj = np.nan

        if num_val_samples > 0:
            val_obj = sum(eval_fn(*val_sampler(i)) for i in xrange(num_val_samples))
            print('minimize_theano_fn::Iteration %d - Validation Objective = %e' % (it, val_obj))

            if val_obj < best_val:
                best_val = val_obj
                best_it = it
                copy_params(params, best_params)

            batch_val_obj = min(val_obj, batch_val_obj)

            if (it + 1) % validation_batch == 0:
                print((step, batch_val_obj, last_batch_val_obj))
                eps = 1e-3;
                # End of validation batch. Check for early stopping.
                if np.isnan(batch_val_obj) or batch_val_obj > (1 + eps) * last_batch_val_obj:

                    # Large increase in error
                    # Try decreasing step size
                    step = step / 2;
                    if step < min_step:
                        print('Reached minimum step after %d iterations' % it)
                        break
                    else:
                        print('Reduced step size to %e at iteration %d' % (step, it))
                        it = best_it
                        copy_params(best_params, params)
                        optimizer.reset_context(context)
                elif batch_val_obj > (1 - eps) * (last_batch_val_obj - 1e-16):
                    # Small change in error. Stop
                    print('Early stopping after %d iterations' % it)
                    break
                else:
                    # Probably can still improve, proceed.
                    last_batch_val_obj = batch_val_obj
                    batch_val_obj = np.inf
                    context = optimizer.get_context()

        if not logger is None:
            logger(it, val_obj)

        it += 1

    if num_val_samples > 0:
        copy_params(best_params, params)


def test_Adam():
    # Test Adam Optimizer       
    n = 1000
    p = 5
    X = np.random.randn(n, p)
    W = np.random.randn(p)
    y = np.dot(X, W)

    Wh = theano.shared(np.zeros(p))
    t_x = T.dmatrix()
    t_yh = T.dot(t_x, Wh)
    t_y = T.dvector()

    t_err = T.sum((t_y - t_yh) ** 2)

    opt = TheanoAdamOptimizer(alpha=1e-2)
    opt.init(t_err, [t_x, t_y], [Wh])

    B = 10
    for t in xrange(1000):
        i = np.random.randint(0, n // B)
        Xi = X[i * B:(i + 1) * B, :]
        yi = y[i * B:(i + 1) * B]
        print(opt.step([Xi, yi]))

    # Test minimize_theano_fn
    print('Testing minimize_theano_fn')
    Wh.set_value(np.zeros(p))

    def log(it, obj):
        if it % 10 == 0:
            print(t_err.eval({t_x: X, t_y: y}))

    sampler = lambda i: [X[i * B:(i + 1) * B, :], y[i * B:(i + 1) * B]]

    minimize_theano_fn(t_err, [t_x, t_y], [Wh], n // B, sampler,
                       num_val_samples=n // B, val_sampler=sampler,
                       initial_step=1.0, max_iterations=50, logger=log)


def test_CG():
    # x dimension needs to be smaller than b
    # Ax=b       
    n = 15  # low rank
    p = 20
    NITER = 4
    A = np.random.randn(p, n)
    A = np.dot(A, A.T)
    b = np.random.randn(p)
    xr = np.random.randn(p)
    bin = np.dot(A, xr)
    # b=bin
    # print (A,b)
    import time

    A0 = np.copy(A)
    C = np.linalg.pinv(A).T
    y = np.dot(b, C)
    # assert np.allclose(np.dot(A, y), b), 'overconstrained system n<p'

    TA = theano.shared(A)
    Tb = theano.shared(b)
    eval = eval_CG(iter=NITER)
    tic = time.time();
    yy_hat = eval(A, b);
    t2 = time.time() - tic;
    tic1 = time.time();
    ny_hat = CGD_num_optimizer(A, b, iter=NITER);
    t3 = time.time() - tic1;
    tic2 = time.time();
    y_hat = TheanoCGDOptimizer(inputs=(TA, Tb)).minimize(iter=NITER).get_value();
    t1 = time.time() - tic2;

    print('y=', y)
    print('yhat took ', t1, y_hat)
    print('yyhat took ', t2, yy_hat)
    print('nyhat took ', t3, ny_hat)

    print('L2 error cgc vs inv: %f err %f' % (np.linalg.norm(y - yy_hat), np.linalg.norm(np.dot(A, yy_hat) - b)))
    print(np.allclose(np.dot(A, yy_hat), b))
    assert np.allclose(np.dot(A, yy_hat), b), 'not sym?'


def test_cg_solve():
    print('test_cg_solve')
    np.random.seed(0)
    d = 5
    A = np.random.rand(d, d)
    A = A + A.T

    b = np.random.rand(d)

    t_A = T.matrix()
    t_b = T.vector()

    x1 = npla.solve(A + np.eye(d), b)
    x2 = cg_solve(t_A, t_b, d, 1.0, False).eval({t_A: A, t_b: b})
    x3 = cg_solve(t_A, t_b, d, 1.0, True).eval({t_A: A, t_b: b})
    assert np.allclose(x1, x2)
    assert np.allclose(x1, x3)


def test_cg_solve_batch():
    print('test_cg_solve_batch')
    np.random.seed(0)

    N = 5
    d = 10
    AA = np.random.rand(N, d, d)
    AA = AA + AA.transpose(0, 2, 1)

    B = np.random.rand(N, d)

    t_AA = T.tensor3()
    t_B = T.matrix()
    t_A = T.matrix()
    t_b = T.vector()

    # Test Exact Solution            
    X1 = np.stack([npla.solve(AA[i] + np.eye(d), B[i]) for i in xrange(N)])
    X2 = np.stack([cg_solve(t_A, t_b, 10, 1.0, False).eval({t_A: AA[i], t_b: B[i]}) for i in xrange(N)])
    X3 = cg_solve_batch(t_AA, t_B, 10, 1.0, False).eval({t_AA: AA, t_B: B})
    X4 = cg_solve_batch(t_AA, t_B, 10, 1.0, True).eval({t_AA: AA, t_B: B})

    assert np.allclose(X1, X2)
    assert np.allclose(X1, X3)
    assert np.allclose(X1, X4)

    # Test Approximate Solution
    X5 = np.stack([cg_solve(t_A, t_b, 5, 1.0, False).eval({t_A: AA[i], t_b: B[i]}) for i in xrange(N)])
    X6 = cg_solve_batch(t_AA, t_B, 5, 1.0, False).eval({t_AA: AA, t_B: B})
    X7 = cg_solve_batch(t_AA, t_B, 5, 1.0, True).eval({t_AA: AA, t_B: B})

    assert np.allclose(X5, X6)
    assert np.allclose(X5, X7)


def test_neumann_inv():
    print('test_neumann_inv')
    np.random.seed(0)
    t_A = T.matrix()
    t_B = neumann_inv(t_A, 50, reg=2.0)

    A = np.random.randn(5, 5)
    B1 = npla.inv(A + 2.0 * np.eye(5))
    B2 = t_B.eval({t_A: A})
    assert np.allclose(B1, B2)


def test_neumann_inv_batch():
    print('test_neumann_inv_batch')
    np.random.seed(0)
    r = 5.0

    t_A = T.matrix()
    t_B = neumann_inv(t_A, 50, reg=r)

    t_AA = T.tensor3()
    t_BB = neumann_inv_batch(t_AA, 50, reg=r)

    AA = np.random.randn(10, 5, 5)
    B1 = np.array([npla.inv(A + r * np.eye(5)) for A in AA])
    B2 = np.array([t_B.eval({t_A: A}) for A in AA])
    B3 = t_BB.eval({t_AA: AA})
    assert np.allclose(B1, B3)


def test_batched_matrix_inverse():
    print('test_batched_matrix_inverse')
    N = 20
    d = 5

    t_X = T.tensor3()
    t_V = T.tensor3()
    t_Y = batched_matrix_inverse(t_X)

    t_i = T.iscalar()
    t_j = T.iscalar()
    t_k = T.iscalar()

    t_R1 = T.Rop(t_Y, t_X, t_V)
    t_g = T.grad(t_Y[t_i, t_j, t_k], t_X)
    f = theano.function([t_i, t_j, t_k, t_X, t_V], (t_g * t_V).sum())

    for i in xrange(5):
        X = np.random.randn(N, d, d)

        # Check value
        Y = t_Y.eval({t_X: X})
        assert np.allclose(Y, npla.inv(X))

        # Check gradient                        
        M = np.random.randn(N, d, d)
        theano.gradient.verify_grad(lambda x: batched_matrix_inverse(x * M).sum(), [X], rng=np.random)

        # Check Rop
        V = np.random.randn(N, d, d)
        R1 = t_R1.eval({t_X: X, t_V: V})
        R2 = np.empty((N, d, d))

        for i in xrange(N):
            for j in xrange(d):
                for k in xrange(d):
                    R2[i, j, k] = f(i, j, k, X, V)

        assert np.allclose(R1, R2)


if __name__ == '__main__':
    np.random.seed(0)

    # test_CG()

    test_batched_matrix_inverse()
    test_cg_solve()
    test_cg_solve_batch()
    test_neumann_inv()
    test_neumann_inv_batch()
