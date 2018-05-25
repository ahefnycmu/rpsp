# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:54:43 2016

@author: ahefny
"""
import numpy as np
import utils.linalg
import utils.feats
import utils.kernel
from collections import Counter

class FeatureExtractor(object):    
    def __init__(self, rng=None):
        self._frozen = False
        self.is_built = False
        self._rng = rng
        self.set_rebuild_option('error')
        
    def set_rebuild_option(self, rebuild):
        assert rebuild in ['rebuild', 'pass', 'error']
        self._rebuild = rebuild
        
    def build(self, all_raw_data):    
        '''
        Build feature extractor given a matrix of all raw data    
        '''
        if self.is_built:
            if self._rebuild == 'error':
                raise Exception('Attempting to rebuild a feature extractor.')
            elif self._rebuild == 'pass':
                return self
        
        self._build(all_raw_data)
        self.is_built = True            
        return self
    
    def _build(self, all_raw_data):
        pass

    def save(self):
        raise NotImplementedError
    
    def load(self, param):
        self._load(param)
        self.is_built = True
    
    def _load(self, param):
        raise NotImplementedError
                    
    def process(self, raw_data):
        '''
        Return the feature matrix for a raw data matrix.
        '''
        is_vec = (len(raw_data.shape) == 1)
        if is_vec:
            raw_data = raw_data.reshape((1,-1))
            
        output = self._process(raw_data)
        
        if is_vec:
            output = output.reshape(-1)
            
        return output

    def _process(self, raw_data):
        return raw_data
        
    def copy(self):
        raise NotImplementedError
                    
def feature_extractor_from_fn(fn):
    extractor = FeatureExtractor()
    extractor._process = fn        
    return extractor
        
class CompositeFeatureExtractor(FeatureExtractor):
    def __init__(self, outer_extractor, inner_extractor):
        super(CompositeFeatureExtractor, self).__init__()
        self._outer_extractor = outer_extractor
        self._inner_extractor = inner_extractor
        
    def _build(self, all_raw_data):        
        self._inner_extractor.build(all_raw_data)
        data = self._inner_extractor._process(all_raw_data)
        self._outer_extractor.build(data)
        
    def _process(self, raw_data):
        data = self._inner_extractor._process(raw_data)
        return self._outer_extractor._process(data)
        
class RFFFeatureExtractor(FeatureExtractor):
    def __init__(self, num_samples=None, bandwidth=None, orth=False, precomputed_V=None, pw=50, rng=None):
        FeatureExtractor.__init__(self, rng=rng)
        self._p = pw
        assert num_samples is not None or precomputed_V is not None
        
        if precomputed_V is None:
            self._num_samples = num_samples
            self._s = bandwidth
            self._frozen = False
            self._orth = orth
        else:
            self._V = precomputed_V
            self._frozen = True
        
    def _build(self, all_raw_data):                 
        d = all_raw_data.shape[1]        
        if self._s is None:
            #self._s = utils.kernel.median_bandwidth(all_raw_data, max=10000, rng = self._rng)
            self._s = utils.kernel.precentile_bandwidth(all_raw_data, p=self._p, max=10000, rng=self._rng)
            
        self._V = utils.kernel.sample_rff(self._num_samples, d, self._s, orth=self._orth, rng=self._rng)                        
        
    def _process(self, raw_data):        
        return utils.kernel.rff(raw_data, self._V) 
    
    def save(self):
        return self._V    
    
    def _load(self, param_v):           
        self._V = param_v        
 
class GramFunction:
    def build(self, all_raw_data):
        pass
    
    def __call__(self,X,Y):
        raise NotImplemented
        
class RBFGramFunction(GramFunction):
    def __init__(self, percentile=50, rng=None):
        self._p = percentile
        self._rng = rng
        
    def build(self, all_raw_data):
        self._s = utils.kernel.median_bandwidth(all_raw_data, self._p, rng=self._rng)
        
    def __call__(self,X,Y):
        return utils.kernel.gram_matrix_rbf(X,Y,self._s)
 
class RBFDiagGramFunction(GramFunction):
    def __init__(self, rng=None):
        self._rng = rng
        
    def build(self, all_raw_data):        
        self._s = utils.kernel.median_diag_bandwidth(all_raw_data, max=1000, rng=self._rng)
        
    def __call__(self, X, Y):
        G = utils.kernel.gram_matrix_diagrbf(X, Y, self._s)                
        return G
                        
class NystromFeatureExtractor(FeatureExtractor):
    def __init__(self, num_samples=1000, max_dim=None, gram_function=None,rng=None):
        FeatureExtractor.__init__(self,rng=rng)
        self._rng = rng
        if gram_function is None:
            gram_function = RBFGramFunction(rng=rng)
        
        self._num_samples = num_samples
        self._max_dim = max_dim
        self._gram_function = gram_function        
        
    def _build(self, all_raw_data):                    
        if isinstance(self._gram_function, GramFunction):
            self._gram_function.build(all_raw_data)
            
        self._f = utils.kernel.nystrom(all_raw_data, self._num_samples,
                                       self._max_dim, self._gram_function,
                                       rng=self._rng)
        
    def _process(self, raw_data):
        return self._f(raw_data)
        
class IndicatorFeatureExtractor(FeatureExtractor):
    def __init__(self):
        FeatureExtractor.__init__(self)
    
    #TODO: Use sparse arrays
    def _build(self, all_raw_data):        
        self._dims = np.max(all_raw_data,axis=0)+1
        self._output_dim = np.product(self._dims,dtype=np.int)        
        
    def _process(self, raw_data):
        # raw_data[0,:] is the highest order
        output = np.zeros((raw_data.shape[0],self._output_dim))
        idx = raw_data[:,-1].copy()
        for i in xrange(self._dims.size-1,0,-1):
            idx += raw_data[:,i-1] * self._dims[i]
            
        for i in xrange(self._output_dim):
            output[np.where(idx==i),i] = 1

        return output
 
class SparseFeatureExtractor(FeatureExtractor):
    '''
    A discrete feature extractor where a small number of combinations can occur.
    '''
    def __init__(self):
        FeatureExtractor.__init__(self)

    def _data2idx(self, raw_data):                
        idx = raw_data[:,-1].copy()
        for i in xrange(raw_data.shape[1]-1,0,-1):
            idx += raw_data[:,i-1] * self._dims[i]
        return idx.astype(np.int)
        
    #TODO: Use sparse arrays
    def _build(self, all_raw_data):                         
        self._dims = np.max(all_raw_data,axis=0)+1
        idx = self._data2idx(all_raw_data)        
        c = Counter(idx)
        self._sparse_idx = dict(zip(c.keys(),xrange(len(c))))
    
    def _process(self, raw_data):
        idx = self._data2idx(raw_data)
        d = len(self._sparse_idx)
        n = len(idx)
        output = np.zeros((n,d))
        
        for i in xrange(n):            
            output[i,self._sparse_idx[idx[i]]] = 1

        return output
                
'''
Feature Wrappers
'''

'''
class ForceBuildFeatureExtractor(FeatureExtractor):
    def __init__(self, base_extractor):
        self._base_extractor = base_extractor
        
    def _build(self, all_raw_data):
        self._base_extractor.build(all_raw_data)
        return self._base_extractor.process(all_raw_data)
        
    def _process(self, raw_data):
        return self._base_extractor.process(raw_data)
''' 
    
class FeatureWrapper(FeatureExtractor):
    def __init__(self, base_extractor):
        FeatureExtractor.__init__(self)
        self._base_extractor = base_extractor
        
    def _build(self, all_raw_data):
        #if not self._base_extractor.is_built:
        self._base_extractor.build(all_raw_data)

class ProjectionFeatureExtractor(FeatureWrapper):
    def __init__(self, base_extractor, projection_matrix):
        FeatureWrapper.__init__(self, base_extractor)        
        self._U_arg = projection_matrix
        
    def _build(self, all_raw_data):
        FeatureWrapper._build(self, all_raw_data)                
        
        if self._U_arg is not None:
            self._U = self._U_arg
        else:            
            self._build_projection(all_raw_data)
            
    def _build_projection(self, all_raw_data):
        self._U = None
        return NotImplemented
        
    def _process(self, raw_data):          
        return utils.linalg.blk_fn_row(lambda s,e: self._base_extractor.process(raw_data[s:e,:]).dot(self._U), raw_data.shape[0]) 
        
    def save(self):
        params = {}
        params['base'] = self._base_extractor.save()
        params['feat_U'] = self._U
        return params

    def _load(self, params):        
        self._U = params['feat_U']
        self._dim = self._U.shape[1]
        self._base_extractor.load(params['base'])        
        
class RandProjFeatureExtractor(ProjectionFeatureExtractor):
    def __init__(self, base_extractor, dim, rng=None):
        ProjectionFeatureExtractor.__init__(self, base_extractor, None)        
        self._base_extractor = base_extractor
        self._dim = dim
        
    def _build_projection(self, all_raw_data):
        #if not self._base_extractor.is_built:        
        self._in_dim = self._base_extractor.process(all_raw_data[:2,:]).shape[1]
        self._U = self.rng.randn(self._in_dim, self._dim)

                                              
class RandPCAFeatureExtractor(ProjectionFeatureExtractor):
    def __init__(self, base_extractor, pca_dim=None, slack=20, precomputed_U=None, rng=None):
        ProjectionFeatureExtractor.__init__(self, base_extractor, None)
        self._U = None
        self._base_extractor = base_extractor
        self._rng = rng
        
        if precomputed_U is None:            
            self._p = pca_dim
            self._slk = slack
            self._frozen = False
        else:
            self._U = precomputed_U
            self._frozen = True
        
    def _build_projection(self, all_raw_data):
        old_U = None
        if self._U is not None:
            old_U = np.copy(self._U)
                                                    
        f = lambda X: self._base_extractor.process(X.T).T
        
        U,_,_ = utils.linalg.rand_svd_f(
            all_raw_data.T, f,
            self._p, slack=self._slk, rng=self._rng)
            
        self._U = U
        if old_U is not None:
            dmin = min([self._U.shape[0],old_U.shape[0]])
            signs =  np.diag(np.dot(self._U[:dmin,:].T, old_U[:dmin,:]))
            dout = signs.shape[0]
            self._U[:,:dout] = self._U[:,:dout] * signs
            assert (np.diag(np.dot(self._U.T, old_U))>=0).all(), 'flipped sign'

class AppendConstant(FeatureWrapper):
    def _process(self, raw_data):
        d = self._base_extractor.process(raw_data[0]).size        
        out = np.ones((raw_data.shape[0], d+1))
        out[:,:-1] = self._base_extractor.process(raw_data)
        return out
    
    def load(self, params):
        self._base_extractor.load(params)
        
    def save(self):
        return self._base_extractor.save()
                                 
'''
Shortcuts to create feature sets
'''        

def create_uniform_featureset(feature_builder):
    return {'obs': feature_builder(), 'act': feature_builder(),
            'fut_obs': feature_builder(), 'fut_act': feature_builder(),
            'past': feature_builder()}
        
def create_RFFPCA_featureset(num_rff_samples, projection_dim, orth=False, pw=50, rng=None):
    D = num_rff_samples
    p = projection_dim
    create_instance = lambda: RandPCAFeatureExtractor(RFFFeatureExtractor(D, orth=orth, pw=pw, rng=rng), p, rng=rng)
    
    return create_uniform_featureset(create_instance)

if __name__ == '__main__':
    A = np.array([[0,1,2,0,2,1], [1,3,2,0,1,1]]).T    
    
    G = np.zeros((6,12))
    for i in xrange(6):
        G[i,A[i,0]*4+A[i,1]] = 1

    f = IndicatorFeatureExtractor()    
    f.build(A)
    F = f.process(A)
    
    assert np.all(F == G)
    