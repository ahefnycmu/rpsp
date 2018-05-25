'''
Created on Dec 15, 2016

@author: zmarinho
'''

import matplotlib
matplotlib.use('Agg') 

import numpy as np
import sys, time, os, math
from IPython import embed
from environments import *
from NN_policies import *
from models import *
from psr_models.covariance_psr import covariancePSR
from psr_models.utils.numeric import RefineModelGD
from policy_learn import learn_policy, learn_model_policy
from psr_models.features.rff_features import RFF_features
from itertools import imap, chain
from psr_models.rff_psr_model import * 
import psr_models.utils.numeric as num   
import matplotlib.pyplot as plt
import matplotlib as mpl

from distutils.dir_util import mkpath
from psr_models.features.psr_features import PSR_features
from psr_models.features.feat_extractor import *
import cPickle as pickle
import psr_models.utils.plot_utils as pu
import scipy
import scipy.linalg
#plt.ion()  

    
    
    
def test_discrete_model(args):
    np.random.seed(100);
    max_traj_length = 200;
    num_trajs = 100;
    #env = GymEnvironment('CartPole-v0');
    #env = partial_obs_Gym_CartPole_Env('CartPole-v0');
    #env = PartiallyObservableEnvironment(GymEnvironment('CartPole-v0'),np.array([0,2,1,3])) #x,theta, x_dot,theta_dot
    env = PartiallyObservableEnvironment(ContinuousEnvironment('CartPole-v0', cartpole_continuous),np.array([0,1,2,3])) #x,theta, x_dot,theta_dot
    env.reset()
    x_dim, _ = env.dimensions
    output_dim= env.action_info[0]
    
    #create model with refine PSRs
    model = BatchFilteringRefineModel(batch_gen=TrainSetGenerator(),val_ratio = 0.7,\
                                       reg=1e-6, rstep=1e-4, a_dim=output_dim,\
                                       n_iter_refine=10);
    model.initialize_model(RFF_features, rff_dim=5000, p=20, past=12, fut=10)
    
   
    #collect initial trajectories and update model
    model_init = ObservableModel(x_dim)
    pi_learn = RNN_Discrete_Policy(x_dim, output_dim, 16)
    
    start_trajs = env.run(model_init, pi_learn, num_trajs, max_traj_length, render=False)
    model.update(start_trajs, 0)
    
    
    #exploration policy
    #pi_explore = RNN_Discrete_Policy(x_dim = model._filtering_model.dim, output_dim=output_dim,nh = 16);

   
    pi_explore = DiscretePolicy(model._filtering_model.dim, output_dim, 1, [16])
    PiUpdator = VR_Reinforce_RNN_PolicyUpdater(policy = pi_explore, 
                            max_traj_length = max_traj_length,
                            num_trajs = num_trajs);

    
    
    learn_model_policy(PiUpdator, model, env, 
                    max_traj_len = max_traj_length, num_trajs = num_trajs, 
                    num_iter = 100)
    
    
    embed()
    
    end_trajs = env.run(model, pi_explore, num_trajs, max_traj_length, render=True)
    return


def wave(L,seq_length, h, filename='examples/data/rff/wave/'):
    mkpath(filename)
    obs = []; act = [];
    try:
        f = open(filename+'waveN%d_l%d_%.2f.pkl'%(L,seq_length,h),'rb')
        obs = pickle.load(f)
        act = pickle.load(f)
        f.close()
    except Exception:
        for i in xrange(L):
            if i%100==0: print (i)
            ts = np.linspace(0.0, (seq_length-1)*h, seq_length)
            u = np.random.rand(seq_length) - 0.5
            ## LEARN from wave
            xs = np.vstack([np.sin(ts*0.5)+0.1*np.cos(ts*0.3)*u,np.sin(ts*0.5)*u+0.1*np.cos(ts*0.3)]).T
            plt.plot(xs[:,0], xs[:,1])
            plt.savefig(filename+'orig_data.pdf')
            obs.append(xs.T)
            act.append(u.reshape(-1,xs.shape[0]))
        f = open(filename+'waveN%d_l%d_%.2f.pkl'%(L,seq_length,h),'wb')
        pickle.dump(obs, f)
        pickle.dump(act, f)
        f.close()
    return obs, act

def ode(L, seq_length, h, filename='examples/data/rff/ode/'):
    from scipy.integrate import odeint
    mkpath(filename)
    obs = []; act = [];
    try:
        f = open(filename+'odeN%d_l%d_%.2f.pkl'%(L,seq_length,h),'rb')
        obs = pickle.load(f)
        act = pickle.load(f)
        f.close()
    except Exception:
        for i in xrange(L):
            if i%100==0: print (i)
            ts = np.linspace(0.0, (seq_length-1)*h, seq_length)
            u = np.random.rand(seq_length) - 0.5 
            ##LEARN ODE
            func = lambda x,t: np.array([x[1]-0.1*np.cos(x[0])*(5*x[0]-4*x[0]**3+x[0]**5)-0.5*np.cos(x[0])*u[int(np.floor(t/float(h)))],
              -65*x[0]+50*x[0]**3-15*x[0]**5-x[1]-100*u[int(np.floor(t/float(h)))]] )
            xs = odeint(func, [0, 0], ts)
            plt.plot(xs[:,0], xs[:,1])
            plt.savefig(filename+'orig_data.pdf')
            obs.append(xs.T)
            act.append(u.reshape(-1,xs.shape[0]))
        f = open(filename+'odeN%d_l%d_%.2f.pkl'%(L,seq_length,h),'wb')
        pickle.dump(obs, f)
        pickle.dump(act, f)
        f.close()
    return obs, act

def circulant(L, seq_length, h, filename='examples/data/rff/circ/'):
    mkpath(filename)
    obs = []; act = [];
    try:
        f = open(filename+'circulantN%d_l%d_%.2f.pkl'%(L,seq_length,h),'rb')
        obs = pickle.load(f)
        act = pickle.load(f)
        f.close()
    except Exception:
        for i in xrange(L):
            if i%100==0: print (i) 
            ##LEARN from circulant matrix
            xs = np.vstack([scipy.linalg.circulant([1.,2.,3.])]*seq_length).T
            u = np.vstack([4.,5.,6.]*seq_length).reshape(-1)

            plt.plot(xs[:,0], xs[:,1])
            plt.savefig(filename+'orig_data.pdf')
            obs.append(xs.T)
            act.append(u.reshape(-1,xs.shape[0]))
        f = open(filename+'circulantN%d_l%d_%.2f.pkl'%(L,seq_length,h),'wb')
        pickle.dump(obs, f)
        pickle.dump(act, f)
        f.close()
    return obs, act

def vary_train_test(xs, u, feat_set, delta=100, testL=100, valL=100, \
                    fut=10, past=20, rdim=20,\
                    rstep=0.01,min_rstep=1e-5, val_batch=5,\
                    reg=1e-6, refine=5 , train=False, method='ode', filename=''):
    ''' list of observations dxN and list of actions daxN'''
    
    L =  len(xs) #delta*5+valL+testL #length of sequence
    trL = L-valL-testL
    size_tr = range(delta,trL+delta,delta)
    Xtest = xs[trL:trL+testL]
    Utest = u[trL:trL+testL]
    Xval = xs[trL+testL:trL+testL+valL]
    Uval = u[trL+testL:trL+testL+valL]
    predictions_1 = []; predictions_2 = [];
    error_1 = []; error_2 = [];
    
    train_fext = PSR_features(feat_set, fut, past, rdim)
    val_fext = PSR_features(feat_set, fut, past, rdim)
    test_fext = PSR_features(feat_set, fut, past, rdim)
    
    psr = covariancePSR(dim=rdim, use_actions=False, reg=reg)
    rpsr = num.RefineModelGD(rstep=rstep, optimizer='sgd', val_batch=val_batch,\
                              refine_adam=False, min_rstep=min_rstep)
    
    print (size_tr)
    for len_seq in size_tr:
        print ('train with L=%d'%len_seq)
        Xtr = xs[:len_seq]
        Utr = u[:len_seq]
        
        #test for PSR
        train_feats, train_data = train_fext.compute_features(Xtr,Utr)
        psr.train(train_fext, train_feats, train_data)
        train_fext.freeze()
        
        val_feats, val_data = val_fext.compute_features(Xval,Uval, base_fext=train_fext)
        test_feats, test_data = test_fext.compute_features(Xtest, Utest, base_fext=train_fext)
        if train:
            tdata = train_data 
            tfeats = train_feats
        else:
            tdata = test_data
            tfeats = test_feats
        pred_1,err_1,states_1 = psr.iterative_test_1s(tdata)

        rpsr.model_refine(psr, train_feats, train_data, n_iter=refine, val_feats=val_feats, val_data=val_data, reg=reg)
        pred_2,err_2,states_2 = psr.iterative_test_1s(tdata)
        error_1.append(err_1)
        error_2.append(err_2)
        predictions_2.append(pred_2)
        predictions_1.append(pred_1)
        train_fext.freeze(False)
    pu.plot_data(tfeats, tdata, psr, predictions_2, error_2, predictions_1, error_1, states_2, filename)
    return



def test_continuous_prediction(args, flname):
    np.random.seed(100);
    valL = 50 #500
    testL = 50 #500
    min_rstep   = 1e-4
    val_batch   = 5
    if args.env=='Swimmer-v1':
        import mujoco_py
        from gym.envs.mujoco import mujoco_env
        env = PartiallyObservableEnvironment( GymEnvironment(args.env, discrete=False),\
                                              np.array([0,1,2,3,4,5])) #dx,dy,dphi,dtheta1,dtheta2
    elif args.env=='CartPole-v0':
        env = PartiallyObservableEnvironment( ContinuousEnvironment(args.env, CartpoleContinuousSimulator()),\
                                              np.array([0,2]))
    env.reset()
    if args.monitor<>'':
        env._base.env.monitor.start(flname+args.monitor)
        env._base.env.monitor.configure(video_callable=lambda count: count%100==0)
#             import gym.wrappers
#             env.env = gym.wrappers.Monitor(env=env.env, directory=flname+args.monitor,\
#                                            force=True, video_callable=lambda count: count%100==0)
    x_dim, a_dim = env.dimensions
    output_dim = env.action_info[0]
    #collect initial trajectories and update model
    model_init = ObservableModel(x_dim)
    
    #pi_learn = RNN_Continuous_Policy(x_dim, a_dim, output_dim, 16)
    pi_learn = ContinuousPolicy(x_dim, a_dim, 0, [16])
    
    trajs = env.run(model_init, pi_learn, args.numtrajs, args.len, render=False)
    obs = [trajs[i].obs.T for i in xrange(len(trajs))]
    act = [trajs[i].act.T for i in xrange(len(trajs))]
    
    if args.fext=='rff':
        feat_set = create_RFFPCA_featureset(args.Hdim, args.dim, args.kw)
    elif args.fext=='nystrom':
        feat_set = create_NystromPCA_featureset(args.Hdim, args.dim, args.kw)
    elif args.fext=='pca':
        feat_set = create_PCA_featureset(args.Hdim, args.dim)
    #indicator_feat_set = create_uniform_featureset(lambda: RandPCAFeatureExtractor(IndicatorFeatureExtractor(), rdim))
    #indicator_feat_set = create_uniform_featureset(IndicatorFeatureExtractor )
    vary_train_test(obs, act, feat_set , args.batch, testL, valL, args.fut, args.past, \
                    args.dim, args.rstep, min_rstep, val_batch, args.reg, args.refine, \
                    int(args.render), args.method, flname)
    return

def test_DS(args, flname):
    h = 0.05
    valL = 50 #500
    testL = 50 #500
    min_rstep   = 1e-5
    val_batch   = 5

    
    method = eval(args.method)
    obs, act = method(args.numtrajs,args.len, h)

    rff_feat_set = create_RFFPCA_featureset(args.Hdim, args.dim)
    #indicator_feat_set = create_uniform_featureset(lambda: RandPCAFeatureExtractor(IndicatorFeatureExtractor(), rdim))
    #indicator_feat_set = create_uniform_featureset(IndicatorFeatureExtractor )

    vary_train_test(obs, act, rff_feat_set , args.batch, testL, valL, args.fut, args.past, \
                args.dim, args.rstep, min_rstep, val_batch, args.reg, args.refine, \
                int(args.render), args.method, flname)
    return

def test_simulated_prediction(args, flname):

    if args.gen=='reward':
        generator = HighRewardTrainSetGenerator(0.3, args.batch) #more peaked in higher rewards
    elif args.gen =='exp':
        generator = ExpTrainSetGenerator(0.2, args.batch) 
    elif args.gen =='boot':
        generator = BootstrapTrainSetGenerator()
    else:
        generator = TrainSetGenerator()
    
    model = BatchFilteringRefineModel(batch_gen=generator,val_ratio = 0.9, \
                                  reg=args.reg, rstep=args.rstep, n_iter_refine=args.refine,\
                                  file=flname, wpred=args.wpred);
    
    if args.env=='Swimmer-v1':
        import mujoco_py
        from gym.envs.mujoco import mujoco_env
        env = PartiallyObservableEnvironment(GymEnvironment(args.env, discrete=False), np.array([0,1]))
    elif args.env=='CartPole-v0':
        env = PartiallyObservableEnvironment(ContinuousEnvironment(args.env, CartpoleContinuousSimulator()),\
                                             np.array([0,2]))
    env.reset()
    if args.monitor<>'':
        env.env.monitor.start(flname+args.monitor)
        env.env.monitor.configure(video_callable=lambda count: count%100==0)

    x_dim, a_dim = env.dimensions
    output_dim = env.action_info[0]
    #collect initial trajectories and update model
    model_init = ObservableModel(x_dim)
    
    #pi_learn = RNN_Continuous_Policy(x_dim, a_dim, output_dim, 16)
    pi_learn = ContinuousPolicy(x_dim, a_dim, 0, [16])
    trajs = env.run(model_init, pi_learn, args.numtrajs, args.len, render=False)
    
    if args.fext=='rff':
        feat_set = create_RFFPCA_featureset(args.Hdim, args.dim, args.kw)
    elif args.fext=='nystrom':
        feat_set = create_NystromPCA_featureset(args.Hdim, args.dim, args.kw)
    elif args.fext=='pca':
        feat_set = create_PCA_featureset(args.Hdim, args.dim)
    

    model.initialize_model(feat_set, p=args.dim, past=args.past, fut=args.fut)
    model.update(trajs)#return states or not?????
    # Restimate the state based in updated model
    for k in xrange(len(trajs)):
        model.filter(trajs[k])
    
    me,se, mfe, sfe = model.prediction_error(trajs, it=0)
    print "prediction error is {} std {} \nfuture error is {} std {} ".format( me,se, mfe, sfe )
    
    pi_final = ContinuousPolicy(model.state_dimension, a_dim, 0 [16])
    env.env.monitor.start(flname+args.monitor+'_finalsim')
    env.env.monitor.configure(video_callable=lambda count: count%1==0)
    env_pred = ContinuousEnvironment('CartPole-v0', PredictedSimulator(model._filtering_model))
    trajs = env_pred.run(model, pi_final, args.numtrajs, args.len, render=True)
    
   
    return
    

def test_psr_feats():
    from scipy.integrate import odeint
    filename = 'results/psr/feats/'
    mkpath(filename)
    fut=10; past=20;
    N = 100; dO = 4; dA=1;
    h=0.05
    num_seqs = 100
    obs = []; act = [];
    f = plt.figure()
    for i in xrange(num_seqs):
        u = np.random.random(N)-0.5
        ts = np.linspace(0.0, (N-1)*h, N)
        func = lambda x,t: np.array([x[1]-0.1*np.cos(x[0])*(5*x[0]-4*x[0]**3+x[0]**5)-0.5*np.cos(x[0])*u[np.floor(t/float(h))],
           -65*x[0]+50*x[0]**3-15*x[0]**5-x[1]-100*u[np.floor(t/float(h))]] )
        xs = odeint(func, [0, 0], ts)
        act.append(u.reshape(1,-1))
        obs.append(xs.T)
        plt.plot(xs[:,0], xs[:,1])
    plt.savefig(filename+'orig_data.pdf')

    rffsamples = 1000
    rdim = 20
    reg = 1e-9
    rstep       = 0.01
    min_rstep   = 1e-5
    refine      = 5
    val_batch   = 5
    #uniform_feat_set = create_uniform_featureset(IndicatorFeatureExtractor)
    rff_feat_set = create_RFFPCA_featureset(rffsamples, rdim)
    psr = covariancePSR(dim=rdim, use_actions=False, reg=reg)
    rpsr = num.RefineModelGD(rstep=rstep, optimizer='sgd', val_batch=val_batch,\
                              refine_adam=False, min_rstep=min_rstep)
    
    
    #test for PSR
    train_fext = PSR_features(rff_feat_set, fut, past, rdim)
    train_feats, train_data = train_fext.compute_features(obs[:50],act[:50])
    train_fext.freeze()
   
    results=psr.train(train_fext, train_feats, train_data)
    
    val_fext = PSR_features(rff_feat_set, fut, past, rdim)
    val_feats, val_data = val_fext.compute_features(obs[50:90], act[50:90], base_fext=train_fext)
    rpsr.model_refine(psr, train_feats, train_data, n_iter=refine, val_feats=val_feats, val_data=val_data, reg=reg)
        
    embed()
    return
    
      
if __name__ == '__main__':
    
    #test_discrete_model(sys.argv[1:])
    test_continuous_model(sys.argv[1:])
    #test_continuous_prediction(sys.argv[1:])
    #test_psr_feats()
    #test_DS(sys.argv[1:])
