# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:54:43 2016

@author: ahefny
"""
import numpy as np
import psr_models.utils.linalg as lg
import psr_models.utils.feats as feat
import psr_models.utils.kernel as kern
from IPython import embed

class FeatureExtractor(object):    
    def __init__(self):
        self._frozen = False
    
    '''
    Build feature extractor given a matrix of all raw data    
    '''
    def build(self, all_raw_data):    
        if not self._frozen:
            self.do_build(all_raw_data)
            
        return self
    
    def do_build(self, all_raw_data):
        pass
                
    '''
    Return the featurematrix for a raw data matrix.
    '''
    def process(self, raw_data):
        return raw_data
    
        '''
    Return the jacobian of featurematrix for a raw data matrix.
    '''
    def process_grad(self, raw_data):
        return raw_data

    def copy(self):
        raise 'Not implemented'
        
    def freeze(self, freeze = True):
        self._frozen = freeze
        
        
    def convert_to_mat(self):
        raise 'Not implemented'
    
    @classmethod
    def from_mat(mat_dict):
        raise 'Not implemented'
    
    
class PowerRFFFeatureExtractor(FeatureExtractor):
    def __init__(self, num_samples=None, s=None, precomputed_V=None, kw=20):
        self._kw = kw
        FeatureExtractor.__init__(self)
        
        if precomputed_V is None:
            self._num_samples = num_samples
            self._s = s
            self._frozen = False
        else:
            self._V = precomputed_V
            self._frozen = True
        
    def do_build(self, all_raw_data):        
        self._V = kern.power_rff(all_raw_data, self._num_samples)                
        
    def process(self, raw_data):
        return kern.rff(self._V, raw_data)
    
    def process_grad(self, raw_data):
        return kern.rff_grad(self._V, raw_data)
        
class RFFFeatureExtractor(FeatureExtractor):
    def __init__(self, num_samples=None, s=None, precomputed_V=None, kw=70):
        self._kw = kw
        FeatureExtractor.__init__(self)
        
        if precomputed_V is None:
            self._num_samples = num_samples
            self._s = s
            self._frozen = False
        else:
            self._V = precomputed_V
            self._frozen = True
        
    def do_build(self, all_raw_data):        
        d = all_raw_data.shape[0]        
        if self._s is None:
            self._s = kern.percentile_bandwidth(all_raw_data, self._kw, max=5000)
            #self._s = kern.percentile_precision(all_raw_data, self._kw, max=5000)
            print 'kernel rff bandwidth: ', self._s
        self._V = kern.sample_rff(self._num_samples, d, self._s)                
        
    def process(self, raw_data):
        return kern.rff(self._V, raw_data)
    
    def process_grad(self, raw_data):
        return kern.rff_grad(self._V, raw_data)
    
    
    
        
class NystromFeatureExtractor(FeatureExtractor):
    def __init__(self, num_samples=1000, bandwidth=None, kw=50):
        FeatureExtractor.__init__(self)
        self._num_samples = num_samples
        self._s = bandwidth
        self._kw = kw
        
    def do_build(self, all_raw_data):
        if self._s is None:
            self._s = kern.percentile_bandwidth(all_raw_data, self._kw, max=5000)
            #self._s = kern.percentile_precision(all_raw_data, self._kw, max=5000)
            print 'kernel nystrom bandwidth: ', self._s
        self._f,Xs,W = kern.nystrom(all_raw_data,self._num_samples,self._s, return_Xs_W=True)
        self._f_grad = kern.nystrom_grad(all_raw_data, W, Xs, self._s)
        
    def process(self, raw_data):
        return self._f(raw_data)
    
    def process_grad(self, raw_data):
        return self._f_grad(raw_data)
        
        
class IndicatorFeatureExtractor(FeatureExtractor):
    def __init__(self):
        FeatureExtractor.__init__(self)
    
    #TODO: Use sparse arrays
    def do_build(self, all_raw_data):        
        self._dims = np.max(all_raw_data,axis=1)+1
        self._output_dim = np.product(self._dims,dtype=np.int)        
        
    def process(self, raw_data):
        # raw_data[0,:] is the highest order
        output = np.zeros((self._output_dim, raw_data.shape[1]))
        idx = raw_data[-1,:][:]
        for i in xrange(self._dims.size-1,0,-1):
            idx = idx + raw_data[i-1,:] * self._dims[i]
            
        for i in xrange(self._output_dim):
            output[i,np.where(idx==i)] = 1

        return output
    
    def process_grad(self, raw_data):
        return 1.0 
 
# Features Adaptors 
class ForceBuildFeatureExtractor(FeatureExtractor):
    def __init__(self, base_extractor):
        self._base_extractor = base_extractor
        
    def build(self, all_raw_data):
        self._base_extractor.build(all_raw_data)
        return self._base_extractor.process(all_raw_data)
        
    def process(self, raw_data):
        return self._base_extractor.process(raw_data)
                               
class RandPCAFeatureExtractor(FeatureExtractor):
    def __init__(self, base_extractor, pca_dim=None, slack=20, precomputed_U=None):
        FeatureExtractor.__init__(self)
        
        self._base_extractor = base_extractor
        
        if precomputed_U is None:            
            self._p = pca_dim
            self._slk = slack
            self._frozen = False
        else:
            self._U = precomputed_U
            self._frozen = True
        
    def do_build(self, all_raw_data):
        self._base_extractor.do_build(all_raw_data)
                
        U,_,_ = lg.rand_svd_f(
            all_raw_data, self._base_extractor.process,
            self._p, slack=self._slk)
            
        self._U = U
        
    def process(self, raw_data):
        return np.dot(self._U.T, self._base_extractor.process(raw_data))      
        
    def process_grad(self, raw_data):
        G = self._base_extractor.process_grad(raw_data)
        assert self._U.shape[0]==G.shape[0], embed() 
        return np.dot(self._U.T, G)   
    
    
class RandCCAFeatureExtractor(FeatureExtractor):
    def __init__(self, base_extractor, cca_dim=None, slack=20, precomputed_U=None):
        FeatureExtractor.__init__(self)
        
        self._base_extractor = base_extractor
        
        if precomputed_U is None:            
            self._p = cca_dim
            self._slk = slack
            self._frozen = False
        else:
            self._U = precomputed_U
            self._frozen = True
        
    def do_build(self, all_raw_data):
        self._base_extractor.do_build(all_raw_data)
                
        
        U,s,V = lg.rand_svd_f(
            all_raw_data, self._base_extractor.process,
            self._p, slack=self._slk)
            
        self._U = U
        self._V = V
        self._s =s
    def process(self, raw_data):
        return np.dot(self._U.T, self._base_extractor.process(raw_data))      
        
    def process_grad(self, raw_data):
        G = self._base_extractor.process_grad(raw_data)
        assert self._U.shape[0]==G.shape[0], embed() 
        return np.dot(self._U.T, G)   
        
               
def create_uniform_featureset(feature_builder):
    return {'obs': feature_builder(), 'act': feature_builder(),
            'fut_obs': feature_builder(), 'fut_act': feature_builder(),
            'past': feature_builder()}
        
def create_RFFPCA_featureset(num_rff_samples, projection_dim, kw=50):
    D = num_rff_samples
    p = projection_dim
    create_instance = lambda: RandPCAFeatureExtractor(RFFFeatureExtractor(D, kw=kw), p)
    
    return create_uniform_featureset(create_instance)

def create_NystromPCA_featureset(num_rff_samples, projection_dim, kw=50):
    D = num_rff_samples
    p = projection_dim
    create_instance = lambda: RandPCAFeatureExtractor(NystromFeatureExtractor(D, kw=kw), p)
    
    return create_uniform_featureset(create_instance)

def create_PCA_featureset( projection_dim, orth=False):
    p = projection_dim
    create_instance = lambda: RandPCAFeatureExtractor(FeatureExtractor(), p)
    return create_uniform_featureset(create_instance)

if __name__ == '__main__':
    A = np.array([[0,1,2,0,2,1], [1,3,2,0,1,1]])    
    
    G = np.zeros((12,6))
    for i in xrange(6):
        G[A[0,i]*4+A[1,i],i] = 1

    f = IndicatorFeatureExtractor()    
    f.build(A)
    F = f.process(A)

    assert np.all(F == G)
    embed()
    