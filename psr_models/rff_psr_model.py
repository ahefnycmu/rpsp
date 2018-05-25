'''
Created on Dec 15, 2016

@author: zmarinho
'''
import numpy as np
import sys, time, os
from IPython import embed
from psr_models.utils.numeric import RefineModelGD
from itertools import imap, chain
from psr_models.utils.feats import ind2onehot, onehot2ind
import psr_models.utils.plot_utils as pu
import matplotlib.pyplot as plt
from psr_models.covariance_psr import covariancePSR
from psr_models.features.psr_features import PSR_features
import psr_models.utils.linalg as lg
from models import *
import matplotlib.cm as cm
from distutils.dir_util import mkpath


class BatchFilteringRefineModel(BatchTrainedFilteringModel):
    def __init__(self, batch_gen=TrainSetGenerator(), model=covariancePSR, refiner=RefineModelGD, \
                 env=None, file='', params={}, a_dim=None,o_dim=None, start_gen=None):
        # initialize only with a_dim, o_dim if indicator representation
        super(BatchFilteringRefineModel, self).__init__( batch_gen=batch_gen )
        self._filtering_model = model
        self.refiner_type = refiner
        self.refiner = None
        self._params = params
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.reset_trajectories()
        self.fname = file
        self._iter = 1
        self._train_data = None
        self._train_feats = None
        self._model = None
        self._env = env
        self._results = []
        self.reward_prev = 0.0
        self.prev_Ur = None
        #self._train_trajs = []
        self._start_gen = start_gen
        
        
    def reset_trajectories(self):
        self._batch_gen._trajs = TrajectoryList(self.a_dim, self.o_dim, [])
        return
        
    def split_validation(self, trajs):
        ''' split dataset into train and validation'''
        #if self._train_trajs is None:
        val_ratio = self._params['valratio']
        n_all = len(trajs)
        n_train = int(np.ceil(n_all*val_ratio))
        
        idx_train = np.random.choice(n_all,n_train,replace=False) 
        idx_val = list( set(np.arange(n_all)) - set(idx_train) )
        
        train = trajs.get(idx_train)
        validation = trajs.get(idx_val)
        if len(validation)==0:
            validation = None
#         else:
#             train = self._train_trajs
#             validation = trajs
        return train, validation

    def initialize_model(self, feat_set, start_trajs, a_dim=None, o_dim=None, seed=100):
        ''' compute features of validation and train trajectories'''
        if a_dim is not None:
            self.a_dim = a_dim
        if o_dim is not None:
            self.o_dim = o_dim
        np.random.seed(seed)
        self._filtering_model = self._filtering_model(feat_set, self._params, env=self._env)
        self._train_fext = PSR_features( feat_set, self._params)
        self._val_fext = PSR_features( feat_set, self._params)
        for t in start_trajs:
            t.policy_grads = np.zeros((t.length, self.state_dimension))
        self._start_filtering_model(start_trajs)
        self.prediction_error(start_trajs, plot=True, it=-1)
        return
        
        
    def refine_model(self, train_feats, train_data, val_trajs):
        if val_trajs is not None:
            print 'val set size:', len(val_trajs), ' shape:',self._train_data.obs.shape[1], ' max traj len ', \
            val_trajs.len_max, ' dataset ', len(self._batch_gen._trajs), ' avg rwd (', np.mean(val_trajs.cum_reward), np.std(val_trajs.cum_reward), ') max-min:', \
            np.max(val_trajs.cum_reward), np.min(val_trajs.cum_reward), np.linalg.norm(np.hstack(val_trajs.observations))

            val_feats, val_data = self._val_fext.compute_features(val_trajs, base_fext=self._train_fext )
            if self._iter>1:
                delta_norm = np.linalg.norm(self.reward_prev,axis=0).mean() - np.linalg.norm(val_data.grads,axis=0).mean()
                print 'Reward grad delta norm', delta_norm, len(np.linalg.norm(val_data.grads,axis=0).nonzero()[0]), np.linalg.norm(val_data.grads,axis=0).max(), np.linalg.norm(val_data.grads,axis=0).min()
            self.reward_prev = val_data.grads[:]
        else:
            val_feats=None; val_data = None;
            if self._iter>1:
                delta_norm = np.linalg.norm(self.reward_prev,axis=0).mean() - np.linalg.norm(train_data.grads,axis=0).mean()
                print 'Reward grad delta norm', delta_norm, len(np.linalg.norm(train_data.grads,axis=0).nonzero()[0]), np.linalg.norm(train_data.grads,axis=0).max(), np.linalg.norm(train_data.grads,axis=0).min()
            self.reward_prev = train_data.grads[:]
        self.refiner = self.refiner_type(rstep=self._params['rstep'], optimizer=self._params['optimizer'],\
                                        val_batch=self._params['valbatch'], refine_adam=self._params['adam'],\
                                        min_rstep=self._params['minrstep'])
        states = self.refiner.model_refine(self._filtering_model, train_feats, train_data, \
                                           val_feats = val_feats, val_data=val_data,\
                                           n_iter=self._params['refine'], reg=self._params['reg'],\
                                           wpred=self._params['wpred'], wnext=self._params['wnext'])
            
        return states
    
    def _train_filtering_model(self, train):
        if self._train_fext._frozen:
            self._train_fext.freeze(False) #build features again
        print 'training set size:', len(train), ' shape:',self._train_data.obs.shape[1], ' max traj len ', \
            train.len_max, ' dataset ', len(self._batch_gen._trajs), ' avg rwd (', np.mean(train.cum_reward), np.std(train.cum_reward), ') max-min:', \
            np.max(train.cum_reward), np.min(train.cum_reward), np.linalg.norm(np.hstack(train.observations))
        results = self._filtering_model.train(self._train_fext, self._train_feats, self._train_data)
        assert results['data']==self._train_data, embed()
        self._train_fext.freeze(True)
        return results['states']

    
    def train(self, trajs, val_ratio=None, n_iter_refine=None):
        if val_ratio is not None:
            self._params['valratio'] = val_ratio
        if n_iter_refine is not None:
            self._params['refine'] = n_iter_refine
        tic = time.time()
        train_trajs, val_trajs = self.split_validation(trajs)
        if self._iter <self._params['ntrain']+1 and len(train_trajs)>0: #train initially only
            self._train_feats, self._train_data = self._train_fext.compute_features(train_trajs)
            states = self._train_filtering_model(train_trajs)
        toc = time.time()
        if self._params['refine']>0:
            states = self.refine_model(self._train_feats, self._train_data, val_trajs)           
            #val_feats, val_data = self._val_fext.compute_features(val_trajs, base_fext=self._train_fext)
            #pred_1,err_1,states_1 = self._filtering_model.iterative_test_1s(val_data)
            #pred_2,err_2,states_2 = self._filtering_model.iterative_test_1s(val_data)
            #pu.plot_data(val_feats, val_data, self._filtering_model, [pred_2], [err_2], [pred_1], [err_1], states_2, self.fname)
        tac = time.time()
        
        print 'train done took psr:%.1f  refine:%.1f secs'%(toc-tic,tac-toc)
        self._iter+= 1
        return states
        
    def _start_filtering_model(self, start_trajs):
        self._trains_trajs = TrajectoryList(start_traj=start_trajs)
        self._train_feats, self._train_data = self._train_fext.compute_features(self._trains_trajs)
        if self._start_gen is not None:
            self._start_gen.update(start_trajs)
            batch = self._start_gen.gen_data()
        else:
            self._batch_gen.update(start_trajs)
            batch = self._batch_gen.gen_data()
        states = self.train(batch)       
        for k in xrange(len(start_trajs)):
                self.filter(start_trajs[k])
        #self.reset_trajectories() #only deletes _trajs
        self._train_fext.freeze(True)
        return states
    
        
    def reset(self, first_observation):
        self.state = self._filtering_model._start
        return self.state
        
    def update_state(self, o, a):
        #test with conditioning on action and measure observation prediction error
        self.state = self._filtering_model.filter(self.state, o, a=a).squeeze()
        return self.state
    
    def save_model(self, filename):
        self._filtering_model.save_model(filename)   
        return
    
    def _get_parameters(self):
        model = self._filtering_model.get_parameters()
        parameters = np.hstack([model['Wex'].reshape(-1, order='F'),\
                                model['Woo'].reshape(-1, order='F')])
        return parameters
        
    def _set_parameters(self, vec):
        model = self._filtering_model.get_parameters()
        Lex = model['Wex'].shape
        Loo = model['Woo'].shape
        assert vec.ndim==1 and (vec.shape==np.prod(Lex) + np.prod(Loo)), 'wrong vector dimension' 
        Wex = vec[:np.prod(Lex)].reshape(Lex, order='F')
        Woo = vec[np.prod(Lex):].reshape(Loo, order='F')
        print 'Wex diff: ', np.linalg.norm(Wex - model['Wex'])
        print 'Woo diff: ', np.linalg.norm(Woo - model['Woo'])
        model['Wex'] = Wex
        model['Woo'] = Woo
        self._filtering_model._set_parameters(model)
        return
        
    
    def prediction_error(self, trajs, plot = True, it=0):
        trajs = TrajectoryList(self.a_dim, self.o_dim, trajs)
        err_trajs = []; ferr_trajs = [];
        err = []; ferr = [];
        actions = trajs.actions
        observations = trajs.observations
        states = trajs.states
        fut = self._filtering_model.feature_extractor._fut
        if plot:
            fig, ax = plt.subplots(3, sharex=True, sharey=False)
        L = 0; pfo=[]; po=[];
        for i in xrange(len(trajs)):
            for j in xrange(trajs[i].length):
                state = states[i][:,j]
                a = actions[i][:,j]
                o = observations[i][:,j]
                po.append(self._filtering_model.predict(state,a=a) )
                err.append(np.linalg.norm(po[-1]-o)/np.linalg.norm(o))
                if j < (trajs[i].length - fut):
                    fa = actions[i][:,j:j+fut].reshape((-1,1), order='F')
                    fo = observations[i][:,j:j+fut].reshape((-1,1), order='F')
                    pfo.append(self._filtering_model.predict_future(state, fa) )
                    fo_err = np.linalg.norm(fo - pfo[-1], 'fro')/ np.linalg.norm(fo)
                    ferr.append(fo_err)
            err_trajs.extend(err)
            ferr_trajs.extend(ferr)
            if plot:
                ax[0].plot(np.arange(len(ferr))+L, ferr, color='g', label='fut_error')
                ax[1].plot(np.arange(len(err))+L, err, color='b', label='next_error')
                ax[2].plot(np.arange(len(pfo))+L, [pfo[k][0,0] for k in xrange(len(pfo))], color='g', label='fut')
                ax[2].plot(np.arange(len(po))+L, [po[k][0,0] for k in xrange(len(po))], color='b', label='next')
                ax[2].plot(np.arange(observations[i].shape[1])+L, observations[i][0,:], color='r', label='true')
                L+= trajs[i].length-1
                ax[0].axvline(x=L, ymin=0, ymax=1, linewidth=0.5, color='k')
                ax[1].axvline(x=L, ymin=0, ymax=1, linewidth=0.5, color='k')
                ax[2].axvline(x=L, ymin=0, ymax=1, linewidth=0.5, color='k')
                if i==0:
                    fig2, ax2 = plt.subplots(observations[0].shape[0], sharex=True, sharey=False)
                    if observations[0].shape[0]==1:
                        ax2=[ax2]
                    colors=['r','b','k','g','y','m','c']
                    
                    for t in xrange(observations[0].shape[0]):
                        ax2[t].plot(np.arange(len(observations[0].T))+L, observations[0][t,:],'--',color=colors[t],label='%d gold'%t)
                        #ax2[t].plot(np.arange(len(observations[0].T)-fut)+L, [po[k][t,0] for k in xrange(len(pfo))],color=colors[t],label='%d futobs'%t)
                        ax2[t].plot(np.arange(len(observations[0].T))+L, [po[k][t,0] for k in xrange(len(po))], color=colors[t],label='%d obs'%t)
                        #embed()
                        ax2[t].legend()
                    fig2.savefig(self.fname+'predobs_%d.pdf'%it)
#                     ffig=plt.figure()
#                     for t in xrange(observations[0].shape[0]):
#                         plt.plot(np.arange(len(observations[0].T))+L, observations[0][t,:],'-',color=colors[t],label='%d gold'%t)
#                     ffig.savefig(self.fname+'test%d.jpg'%it)
#                     ffig=plt.figure()
#                     for t in xrange(observations[0].shape[0]):
#                         plt.plot(np.arange(len(observations[0].T))+L, [po[k][t,0] for k in xrange(len(po))],'-',color=colors[t],label='%d gold'%t)
#                     ffig.savefig(self.fname+'pred%d.jpg'%it)
                
            err = []; ferr = []; pfo = []; po = [];
            if (i>10) and plot:
                fig.subplots_adjust(hspace=0)
                plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
                ax[0].set_title('next vs future prediction error(top) pred (bottom)')
                #ax[0].legend()
                #ax[1].legend()
                #ax[0].set_ylim([0, np.mean(ferr)+np.std(ferr)])
                #ax[1].set_ylim([0,np.mean(err)+np.std(err)])
                #ax[2].set_ylim([0,np.mean(po)+np.std(po)])
                ax[2].set_xlim([0,200])
                fig.savefig(self.fname+'fut_next_%d.pdf'%it)
                plot = False
        if False:
            Uext,_,_ = lg.rand_svd_f(self._filtering_model._W_s2ext, k=2)
            Uoo,_,_ = lg.rand_svd_f(self._filtering_model._W_s2oo, k=2)
            nmax = np.max([Uext.shape[0], Uoo.shape[0]])
            colors = (cm.gray(np.linspace(0, 1, nmax)))
            f1 = plt.figure()
            plt.scatter(Uext[:,0], Uext[:,1], color=colors[:Uext.shape[0]])
            f1.savefig(self.fname+'Uext_%d.pdf'%it)
            f2 = plt.figure()
            plt.scatter(Uoo[:,0], Uoo[:,1], color=colors[:Uoo.shape[0]])
            f2.savefig(self.fname+'Uoo_%d.pdf'%it)   
        f3, ax3  = plt.subplots(nrows=2, sharex=True, sharey=False)
        m = np.mean([np.sum(t.rewards) for t in trajs]); s = np.std([np.sum(t.rewards) for t in trajs]);
        me = np.mean(err_trajs); se = np.std(err_trajs);
        mfe= np.mean(ferr_trajs); sfe = np.std(ferr_trajs);
        print "future prediction error {} std {}.\nnext prediction error {} std{}.".format(mfe,sfe,me,se)
        self._results.append([m,s,me,se,mfe,sfe])
        m,s,me,se,mfe,sfe = zip(*self._results)
        ts = range(len(trajs),(len(self._results)+1)*len(trajs),len(trajs))
        ax3[0].errorbar(ts, m, yerr=[s, s], fmt='o', ecolor='k', capthick=2)
        ax3[0].plot(ts,m,'-k')
        ax3[0].set_ylabel('avg. reward')
        ax3[0].set_ylim([np.mean(m)-2*np.mean(s), np.mean(m)+2*np.mean(s)])
        ax3[1].errorbar(ts, me, yerr=[se, se], fmt='o', ecolor='b', capthick=2, label='next')
        ax3[1].plot(ts,me,'-b')
        ax3[1].errorbar(ts, mfe, yerr=[sfe, sfe], fmt='o', ecolor='g', capthick=2, label='future')
        ax3[1].plot(ts,mfe,'-g')
        #ax3[1].set_ylim([np.mean(me)-2*np.mean(se), np.mean(me)+2*np.mean(se)])
        #ax3[1].set_xlim([0, ax[1].get_xlim()[1]+200])
        ax3[1].legend()
        ax3[1].set_ylabel('avg. pred. error')
        f3.savefig(self.fname+'/avg_iter.jpg')
        return
                    
    @property
    def state_dimension(self):
        return self._filtering_model.get_dim()
    
    


