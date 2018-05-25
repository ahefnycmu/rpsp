# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:53:31 2016

@author: zmarinho
"""
import sys
import pdb
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

import os
from distutils.dir_util import mkpath

def get_filename(file, algorithm, nclusters, sparse, window, testindex, nsteps):
    params = {'algorithm':algorithm, 'nclusters':nclusters,'sparse':sparse, 'window':window, 'testindex':testindex, 'nsteps':nsteps}
    filename = file+'_'.join([str(k)+'_'+str(v) for k, v in sorted(params.iteritems())])
    mkpath(filename)
    return filename


def read_data(filepath, name=None,delim=' '):
    filein = open(filepath+'/'+name, 'r')
    data = [];
    for line in filein.readlines():
        line = line.rstrip()
        entries = line.split(delim)
        data.append(entries)
    return data
    
def read_matrix(filepath, name=None, delim=' '):
    filein = open(filepath+'/'+name, 'r')
    data = [];
    for line in filein.readlines():
        line = line.rstrip()
        entries = line.split(delim)
        cols = np.asarray(entries, dtype=float)
        data.append(cols)
    return np.asarray(data, dtype=float)
    
    
def center_data(data):
    dN = np.argmax(data.shape)
    N = data.shape[dN]    
    mean = np.mean(data, axis=dN)
    centered_data = data - np.tile(mean,(N,1)).T;
    assert np.max(np.mean(centered_data,axis=dN))<1e-3, pdb.set_trace()
    return centered_data, mean
    
def matrix2lists( A, dim, N=0, L=0):
    ''' get matrix of trajectories rows are samples columns timesteps\
     dim is dimension N number of sequences and L length of sequence \
     A is either dNxL or dxNL'''
    if N==0:
        N = np.divide(A.shape[0], dim) #number of trajectories
    if L==0:
        L = np.divide(A.shape[1], N) #length
    seqs =[]
    for i in xrange(N):
        if A.shape[0]>A.shape[1]:
            seqs.append(A[i*dim:(i+1)*dim, :])
        else:
            seqs.append(A[:dim, i*L:(i+1)*L])
        
    return seqs

def save_data(filename, data):
    f = open(filename, 'wb')
    for i in xrange(len(data)):
        pickle.dump(data[i], f) 
    f.close()
    return


def hungarian(M):
    '''run hungarian method to find ordering of '''
    from munkres import Munkres
    m = Munkres()
    ind = m.compute(M)
    return ind

def bestMap(L1, L2):
    '''permute labels of L2 to match L1 as good as possible'''
    print 'best Map'
    l1= L1[:]
    l2 = L2[:]
    label1 = np.unique(l1)
    nclass1 = label1.shape[0]
    label2 = np.unique(l2)
    nclass2 = label2.shape[0]
    nclass = np.max([nclass1,nclass2])
    G = np.zeros((nclass2,nclass1),dtype=float)
    for i in xrange(nclass1):
        for j in xrange(nclass2):
            G[j,i] = np.sum((L1==label1[i]) * (L2==label2[j]))
    GM = -G if  (nclass2 <= nclass1) else -G.T
    c = hungarian(GM)
    newL2 = np.zeros(L2.shape, dtype=int)
    for i in xrange(nclass2):
        newL2[L2==label2[c[i][0]]] = label1[c[i][1]] #column is input output is row group assignment
    return newL2

def bestAssign(L1, L2):
    '''assign and permute labels of L2 to match L1 as good as possible'''
    print 'best Assign'
    l1= L1[:]
    l2 = L2[:]
    label1 = np.unique(l1)
    nclass1 = label1.shape[0]
    label2 = np.unique(l2)
    nclass2 = label2.shape[0]
    nclass = np.max([nclass1,nclass2])
    G = np.zeros((nclass2,nclass1),dtype=float)
    for i in xrange(nclass1):
        for j in xrange(nclass2):
            G[j,i] = np.sum((L1==label1[i]) * (L2==label2[j]))
    newlabels2 = G.argmax(1)
    newL2 = newlabels2[l2]
    labels = bestMap(L1, newL2)
    return labels


def get_data(datapath, timestamp, sparse, timewindow):

    files = []
    files = [datapath+f for f in os.listdir(datapath) \
             if os.path.isfile(datapath+f) and f.endswith('.csv')]

    datas = []
    timespan = []
 
    for f in files[:-4]: 
        print "loading... ", f 
        data = np.loadtxt(open(f))
        data = data[3:data.shape[0]-3, :]
        tspan = np.array(range(data.shape[0])).reshape((data.shape[0],-1))
        if timestamp:
            data = np.concatenate(
                (data, tspan), axis = 1)
        tsparse = np.arange(1, data.shape[0], sparse)
        data = data[tsparse, :]     
        datas.append(data)
        timespan.extend(tspan[tsparse])

    timespan = np.array(timespan)
    data = np.concatenate(datas, axis=0)
    # test_traj
    test_trajs = datas[-4:-1]
    test_traj_tspan = tspan[tsparse]
    # mean_traj = np.array(datas).mean(axis = 0, keepdims=True)
    # mean_traj = mean_traj[0]

    # normalize per feature 
    import scipy.cluster
    std =  data.std(axis=0,keepdims=True)
    std[abs(std) < 1e-5] = 1.0
    mean = data.mean(axis=0, keepdims=True)
    data = (data-mean)/std

    test_trajs = [(traj - mean)/std for traj in test_trajs]

    if timewindow > 1: 
        stacked_data = np.empty((data.shape[0]-timewindow+1, data.shape[1]*timewindow))
        for i in xrange(data.shape[0]-timewindow+1):
            window = data[i:i+timewindow,:]
            stacked_data[i] = np.squeeze(window.reshape((-1,1)))
        data = stacked_data
        timespan = timespan[:data.shape[0],:]

    return data, timespan, test_trajs, np.array(test_traj_tspan)


def get_data_list(datapath, timestamp, sparse, timewindow):

    files = []
    files = [datapath+f for f in os.listdir(datapath) \
             if os.path.isfile(datapath+f) and f.endswith('.csv')]

    datas = []
    timespan = []
 
    for f in files[:-4]: 
        print "loading... ", f 
        data = np.loadtxt(open(f))
        data = data[3:data.shape[0]-3, :]
        tspan = np.array(range(data.shape[0])).reshape((data.shape[0],-1))
        if timestamp:
            data = np.concatenate(
                (data, tspan), axis = 1)
        tsparse = np.arange(1, data.shape[0], sparse)
        data = data[tsparse, :]     
        datas.append(data)
        timespan.extend(tspan[tsparse])

    timespan = np.array(timespan)
    data = np.concatenate(datas, axis=0)
    # test_traj
    test_trajs = datas[-4:-1]
    test_traj_tspan = tspan[tsparse]
    # mean_traj = np.array(datas).mean(axis = 0, keepdims=True)
    # mean_traj = mean_traj[0]

    # normalize per feature 
    import scipy.cluster
    std =  data.std(axis=0,keepdims=True)
    std[abs(std) < 1e-5] = 1.0
    mean = data.mean(axis=0, keepdims=True)
    data = (data-mean)/std
    datasn = [(datas[i]-mean)/std for i in xrange(len(datas))]
    test_trajs = [(traj - mean)/std for traj in test_trajs]


    return datasn, timespan, test_trajs, np.array(test_traj_tspan)


def get_cylinder_pushing_data(datapath, sparse=1, testSize=10, trainSize=100):
    files = []
    files = [datapath+f for f in os.listdir(datapath) \
             if os.path.isfile(datapath+f) and f.endswith('.pkl')]

    datas = []
    timespan = []
    for i in xrange(testSize+trainSize):
        f = files[i] 
        print("loading... ", f) 
        fpkl = open(f, 'rb')
        data = pickle.load(fpkl)
        rP = data["robotPose"]
        rP = np.concatenate([np.array(val,dtype=float)[None,:] for val in rP.values()], axis=0 )
        rV = data["robotVelocity"]
        rV = np.concatenate([np.array(val,dtype=float)[None,:] for val in rV.values()], axis=0 )
        oP = data["objectPose"]
        oP = np.concatenate([np.array(val,dtype=float)[None,:] for val in oP.values()], axis=0 )
        oV = data["objectVelocity"]
        oV = np.concatenate([np.array(val,dtype=float)[None,:] for val in oV.values()], axis=0 )
        contact = data["touching"]
        contact = np.concatenate([np.array([val],dtype=float)[None,:] for val in contact.values()], axis=0 )
        #tstep = data["timestep"]

        traj = np.concatenate([oP,oV, rP, rV, contact], axis=1)
        tspan = np.array(range(traj.shape[0])).reshape((traj.shape[0],-1))
        obs_dim = np.array([0,1,2,3,4,5])
        action_dim = np.array([6,7,8,9,10,11,12])
        data = traj[::sparse, :]     
        datas.append(traj)
        timespan.append(tspan[::sparse])
       
    timespan = np.array(timespan)
    all_data = np.concatenate(datas[:trainSize], axis=0)
    std =  all_data.std(axis=0,keepdims=True)
    std[abs(std) < 1e-5] = 1.0
    mean = all_data.mean(axis=0, keepdims=True)

    trajs =  [(traj - mean)/(std) for traj in datas[:trainSize+testSize]]
    # divide
    train_trajs = trajs[:trainSize]
    test_trajs = trajs[trainSize:testSize+trainSize]
    test_traj_tspan = tspan[::sparse]
    

    return train_trajs, timespan, test_trajs, np.array(test_traj_tspan), obs_dim, action_dim


def append_trajectories(trajectories, action_dim, size_end, labels=None):
    ''' add stop symbol and pad trajectory of hankel size'''
    print('process trajectories')
    if (labels<>None):
        concat_labels = np.hstack(labels)
        nlabels = np.unique(concat_labels)
        n_label_end = np.max(nlabels)+1 
    
    #add only at start of seq 0 start/stop symbol
    start = trajectories[0][None:1,:]
    start[:,action_dim] = 0.0
    trajs = [ np.vstack([ np.vstack([start]*size_end), trajectories[0] ]) ]
    [trajs.append(trajectories[i+1]) for i in xrange(len(trajectories)-1)]
    if labels<>None:
        lab = [np.hstack([ np.asarray([n_label_end]*size_end, dtype=float), labels[0] ])] 
        [lab.append(labels[i+1]) for i in xrange(len(labels)-1)]
    
    for i in xrange(len(trajectories)):
        end = trajectories[i][-1:None,:]
        end[:,action_dim] = 0.0
        trajs[i] = np.vstack([trajs[i], np.vstack([end]*size_end)])
        if labels<>None:
            lab[i] = np.hstack([ lab[i], np.asarray([n_label_end]*size_end, dtype=float)])
    concat_trajs = np.concatenate(trajs, axis=0)
    concat_labels = np.concatenate(lab, axis=0).reshape(1,-1) 
    return concat_trajs, concat_labels

def process_trajectories_unsup(trajs, obs_dim, action_dim, send, labels=None):
    print('process trajectories')
    concat_trajs,concat_labels = append_trajectories(trajs, action_dim, send, labels=labels)
    obs = concat_trajs[:,obs_dim]
    actions = concat_trajs[:,action_dim]
    return obs.T, actions.T, concat_labels

def process_trajectories(trajs, labels, obs_dim, action_dim, send):
    print('process trajectories')
    assert len(labels)==len(trajs), 'mismatch size labels and trajs!'
    traj,lab = append_trajectories(trajs, obs_dim, action_dim, send, labels)
    observations, actions = build_coo_outer_matrix(traj, lab, obs_dim, action_dim)
    return observations.T, actions.T

