
import matplotlib.pyplot as plt
from psr_models.utils.linalg import dataProjection
import numpy as np
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import cPickle as pickle
fig_number = 0 
colors = np.array([x for x in 'bgrcmyk'])
colors = np.hstack([colors] * 20)
# colors = np.random.random((nclusters,3))


colors2 = ['r','b','g','m','k','c','y']
def plot_2PCs(X, cat, true_cat, title='',filename='data/'): 
    NClass = np.max([len(np.unique(cat)[:]),len(np.unique(true_cat)[:])])
    plt.figure()
    plt.subplot(211)
    Xp = dataProjection(X,2,'PCA')
    Xp = Xp.T
    #rgb = plt.cm.inferno(np.arange(NClass))
    rgb = plt.cm.get_cmap('plasma')
    cind = np.linspace(0,len(rgb.colors)-1,NClass, dtype=int)
    rgbc = np.asarray(rgb.colors)[cind]
    true_cat = np.asarray(true_cat, dtype=int).squeeze()
    cat = np.asarray(cat, dtype=int).squeeze()
    for i in xrange(NClass):
        class_i = np.asarray(true_cat==i,dtype=bool).squeeze()
        plt.scatter(Xp[class_i,0],Xp[class_i,1], color=rgbc[i], label=str(i))
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.title(title+'true')
    plt.legend(loc=4, frameon=False,labelspacing=0.1)
    plt.subplot(212)
    for i in xrange(NClass):
        class_i = np.asarray(cat==i,dtype=bool).squeeze()
        plt.scatter(Xp[class_i,0],Xp[class_i,1], color=rgbc[i], label=str(i))
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.legend(loc=4, frameon=False,labelspacing=0.1)
    plt.title(title+'predicted')
    plt.draw()
    plt.savefig('%s_%ddim_%dsize.png'%(filename,2,X.shape[1]), dpi=300)
    return

def plot_PCs(X, cat, title='', filename='data/',d=2): 
    ''' plot 2 last principal components X is dxN'''
    cat = np.asarray(cat, dtype=int).squeeze()
    NClass = len(np.unique(cat)[:])
    plt.figure()
    plt.subplot(211)
    Xp = dataProjection(X,d,'PCA')
    Xp = Xp.T
    rgb = plt.cm.get_cmap('plasma')
    cind = np.linspace(0,len(rgb.colors)-1,NClass, dtype=int)
    rgbc = np.asarray(rgb.colors)[cind]
    for i in xrange(NClass):
        class_i = np.asarray(cat==i,dtype=bool).squeeze()
        plt.scatter(Xp[class_i,-2],Xp[class_i,-1], color=rgbc[i], label=str(i))
    plt.ylabel('PC2')
    plt.xlabel('PC1')
    #plt.legend(loc=2, frameon=False,labelspacing=0.1)
    plt.title(title)
    plt.draw()
    plt.savefig('%s_%ddim_%dsize.png'%(filename,2,X.shape[1]), dpi=300)
    return

def plot_modes(pred1,pred2,pred3, ylabel='',label1='',label2='',label3='', title='', filename='', fig=None):
    t1 = np.arange(0,pred1.shape[0],1)
    t2 = np.arange(0,pred2.shape[0],1)
    t3 = np.arange(0,pred3.shape[0],1)
    if fig==None:
        plt.figure()
    #plt.subplot(311)
    plt.plot(t1, pred1, color='b', label=label1)
    plt.plot(t3, pred3, color='r', label=label3)
    plt.plot(t2, pred2, color='g', label=label2)
    plt.ylabel(ylabel)
    plt.xlabel('iterations')
    plt.title(title)
    plt.legend(loc=1,ncol=3)
    plt.show()
    if filename<>'':
        plt.savefig(filename+'.jpg')
 
 
def plot_minmax(data, sparse=1, filename='results/min_max.jpg'):
    ''' data covariates x num samples '''
    d = len(data)
    embed()
    f, ax  = plt.subplots(nrows=1, sharex=True, sharey=True)
    ts = range(0,len(data[0]),sparse)
    for it in xrange(d):
        ax[it].errorbar(ts, data[it][ts].mean(), yerr=[data[it][ts].min(), data[it][ts].max()],
            fmt='o', ecolor='g', capthick=2)
    plt.savefig(filename)
    return

def plot_results(results, batch=500, filename='results/'):
    ''' data covariates x num samples '''
    d = len(results)
    
    f, ax  = plt.subplots(nrows=2, sharex=True, sharey=False)
    ts = range(batch,(d+1)*batch,batch)
    m,s,me,se,mfe,sfe = zip(*results)
    ax[0].errorbar(ts, m, yerr=[s, s], fmt='o', ecolor='k', capthick=2)
    ax[0].set_ylabel('avg. reward')
    ax[0].set_ylim([np.mean(m)-np.mean(s), np.mean(m)+np.mean(s)])
    ax[1].errorbar(ts, me, yerr=[se, se], fmt='o', ecolor='b', capthick=2, label='next')
    ax[1].plot(ts,me,'-b')
    ax[1].errorbar(ts, mfe, yerr=[sfe, sfe], fmt='o', ecolor='g', capthick=2, label='future')
    ax[1].plot(ts,mfe,'-g')
    ax[1].set_ylim([np.mean(me)-np.mean(se), np.mean(me)+np.mean(se)])
    ax[1].set_xlim([0, ax[1].get_xlim()[1]+200])
    ax[1].legend()
    ax[1].set_ylabel('avg. pred. error')
    plt.savefig(filename+'/avg_iter_%d.jpg'%batch)
    return

def plot_batch(data, batch=500, filename='results/min_max_iter'):
    ''' data covariates x num samples '''
    f = plt.figure()
    ts = range(0,len(data),batch)
    d = len(ts)-1
    avg = []; min = []; max = [];
    for it in xrange(d):
        trajs = data[ts[it]:ts[it+1]]
        dit = [np.sum(t) for t in trajs]
        avg.append(np.mean(dit) )
        min.append(avg[it]-np.min(dit))
        max.append(np.max(dit)-avg[it])
    plt.errorbar(ts[:-1], avg, yerr=[min, max],
            fmt='o', ecolor='g', capthick=2)
    plt.plot(ts[:-1], avg, color='b')
    plt.savefig(filename+'/minmax_batch%d.jpg'%batch)
    return

def plot_trajs(data, batch=500, filename='results/min_max_iter'):
    ''' data covariates x num samples '''
    f = plt.figure()
    ts = range(0,len(data)*batch,batch)
    d = len(ts)-1
    avg = []; min = []; max = [];
    for it in xrange(d):
        trajs = data[it]
        dit = [np.sum(t) for t in trajs]
        avg.append(np.mean(dit) )
        min.append(avg[it]-np.min(dit))
        max.append(np.max(dit)-avg[it])
    plt.errorbar(ts[1:], avg, yerr=[min, max],
            fmt='o', ecolor='g', capthick=2)
    plt.plot(ts[1:], avg, color='b')
    plt.ylim([-10,plt.ylim()[1]])
    plt.savefig(filename+'/minmax_batch%d.jpg'%batch)
    return


def plot_boxes(data, batch=500, filename='results/box_iter'):
    ''' data covariates x num samples '''
    f = plt.figure()
    ts = range(0,len(data),batch)
    d = len(ts)-1
    sums = [];
    for it in xrange(d):
        trajs = data[ts[it]:ts[it+1]]
        dit = [np.sum(t) for t in trajs]
        sums.append(dit )
    plt.boxplot(sums,1)
    plt.xticks(np.arange(len(ts)), ts)
    #plt.ylim([])
    plt.ylabel('batch rewards')
    plt.xlabel('training trajectories')
    plt.savefig(filename+'/box_batch%d.jpg'%batch)
    return

 
def plot_predictions( pred1, pred2, pred3, err1, err2, err3, title='',\
                       label1='', label2='', label3='', filename=''):
    t = np.arange(0,pred1.shape[0],1)
    t2 = np.arange(0,pred1.shape[2],1)
    plt.figure()
    plt.subplot(311)
    if pred1.shape[1]==1:
        plt.plot(t, pred1[:,0,0], color='b', label=label1)
        plt.plot(t[:], pred3[0,:], color='r', label=label3)
        plt.plot(t, pred2[:,0,0], color='g', label=label2)
    else:
        plt.plot(np.mean(pred1[:,0,:],1), np.mean(pred1[:,1,:],1), color='b', label=label1)
        plt.plot(pred3[0,:], pred3[1,:], color='r', label=label3)
        plt.plot(np.mean(pred2[:,0,:],1), np.mean(pred2[:,1,:],1), color='g', label=label2)
    plt.ylabel('prediction')
    plt.xlabel('time')
    plt.title(title)
    #plt.legend(loc=4, frameon=False,labelspacing=0.1)
    plt.subplot(312)
    plt.plot(t, np.mean(err1,1), 'b',label=label1)
    plt.plot(t, np.mean(err3,1), 'r',label=label3)
    plt.plot(t, np.mean(err2,1), 'g', label=label2)      
    plt.ylabel('error')
    plt.xlabel('time')
    plt.yscale('log')
    #plt.legend(loc=2, frameon=False,labelspacing=0.1)
    plt.subplot(313)
    plt.plot(t2, np.mean(err1,0), 'b',label=label1)
    plt.plot(t2, np.mean(err3,0), 'r', label=label3)
    plt.plot(t2, np.mean(err2,0), 'g',label=label2)
    plt.legend(loc=2, frameon=False,labelspacing=0.1)
    plt.ylabel('error')
    plt.xlabel('horizon')
    #plt.yscale('log')
    plt.show()
    plt.savefig(filename+'.jpg')
    return


def plot_vec(vec):
    plt.figure()
    plt.plot(np.arange(vec.shape[0]),vec)
    plt.draw()
    return

def plot_matrix(A):
    plt.figure()
    plt.pcolor(A)
    plt.show()
    embed()
    return

def plot_array(arraylist, colorshape=['r--']):
    plt.figure(1)
    plt.subplot(111)
    for i in xrange(len(arraylist)):
        d = arraylist[i].shape[0]
        t = np.arange(0., d[-1], 1.0)
        plt.plot(t, arraylist[i], colorshape[i])
    plt.show()
    return

def get_figure_number():
    global fig_number
    fig_number += 1
    print "fig {} created".format(fig_number)
    return fig_number


def draw_traj(timespan, traj, labels, title):

    print title

    fig = plt.figure(get_figure_number())
    fig.clf()
    fig.set_tight_layout(True)
    ylabels = ['x','y','w']*4
    subfign = 1
    for i in xrange(3):
        subfig = int("23"+str(subfign))
        subfign += 1
        ax = fig.add_subplot(subfig, projection='3d')
        ax.scatter3D(timespan, traj[:,i], traj[:, i+3],
                     color=colors[labels].tolist(), s=1)
        ax.set_xlabel('t')
        ax.set_ylabel(ylabels[i])
        ax.set_title('obj')
    
    for i in xrange(6, 9, 1):
        subfig = int("23"+str(subfign))
        subfign += 1
        ax = fig.add_subplot(subfig, projection='3d')
        ax.scatter3D(timespan, traj[:,i], traj[:, i+3],
                     color=colors[labels].tolist(), s=1)
        ax.set_xlabel('t')
        ax.set_ylabel(ylabels[i])
        ax.set_title('robot')

    fig.suptitle(title)


def draw_paired_traj(timespan, paired_traj, labels, title):

    print title 

    fig = plt.figure(get_figure_number())
    fig.clf()
    fig.set_tight_layout(True)
    ylabels = ['x','y','w']*4

    traj1 = paired_traj[0]
    traj2 = paired_traj[1]
    subfign = 1
    for i in xrange(3):
        subfig = int("23"+str(subfign))
        subfign += 1
        ax = fig.add_subplot(subfig, projection='3d')
        ax.scatter3D(timespan, traj1[:,i], traj1[:, i+3],
                     color=colors[labels].tolist(), s=1,
                     marker='o')
        ax.scatter3D(timespan, traj2[:,i], traj2[:, i+3],
                     color=colors[labels].tolist(), s=1,
                     marker='*', alpha=0.75)
        ax.set_xlabel('t')
        ax.set_ylabel(ylabels[i])
        ax.set_title('obj')
    
    for i in xrange(6, 9, 1):
        subfig = int("23"+str(subfign))
        subfign += 1
        ax = fig.add_subplot(subfig, projection='3d')
        ax.scatter3D(timespan, traj1[:,i], traj1[:, i+3],
                     color=colors[labels].tolist(), s=1,
                     marker='o',)
        ax.scatter3D(timespan, traj2[:,i], traj2[:, i+3],
                     color=colors[labels].tolist(), s=1,
                     marker='*', alpha=0.3)
        ax.set_xlabel('t')
        ax.set_ylabel(ylabels[i])
        ax.set_title('robot')

    fig.suptitle(title)



def draw_embedding(timespan, z_traj, cluster_ids, nclusters, title, linestyle='-'):
    print title

    fig = plt.figure(get_figure_number())
    fig.clf()
    fig.set_tight_layout(True)

    for i in xrange(nclusters):
        subfig = "1"+str(nclusters)+str(i+1)
        ax = fig.add_subplot(int(subfig), projection='3d')
        ax.plot(timespan[cluster_ids==i],
                z_traj[cluster_ids==i,0],
                z_traj[cluster_ids==i,1],
                color=colors[i], marker='o', markerfacecolor=colors[i], 
                markeredgecolor=colors[i],
                markersize=1, linestyle=linestyle)

        ax.set_title("cluster " + str(i))
    fig.suptitle(title)
    plt.show

def draw_paired_embedding(timespan, paired_z_traj, cluster_ids, nclusters, title, linestyle='-'):
    print title


    fig = plt.figure(get_figure_number())
    fig.clf()
    fig.set_tight_layout(True)

    z_traj1 = paired_z_traj[0]
    z_traj2 = paired_z_traj[1]

    for i in xrange(nclusters):
        subfig = "1"+str(nclusters)+str(i+1)
        ax = fig.add_subplot(int(subfig), projection='3d')
        ax.plot(timespan[cluster_ids==i],
                z_traj1[cluster_ids==i,0],
                z_traj1[cluster_ids==i,1],
                color=colors[i], marker='o', markerfacecolor=colors[i], 
                markeredgecolor=colors[i],
                markersize=1, linestyle=linestyle)

        ax.plot(timespan[cluster_ids==i],
                z_traj2[cluster_ids==i,0],
                z_traj2[cluster_ids==i,1],
                color=colors[i], alpha=0.35, marker='*', markerfacecolor=colors[i], 
                markeredgecolor=colors[i],
                markersize=1, linestyle=linestyle)

        ax.set_title("cluster " + str(i))
    fig.suptitle(title)
    plt.show()
    

def plot_all_results( results, clusters, filename, save=True):
    embed()
    Nlabels = len(results) # for extra weights
    N = len(results[0]['error'])
    len_max = np.max([results[0]['label_error'][i].shape[0] for i in xrange(N)])
    t_len = np.arange(len_max)
    #extract meaningful analysis
    label_tH_errors={}
    label_next_errors={}
    obs_tH_errors={}
    obs_next_errors={}
    
    all_label_tH_errors=np.zeros((N, len_max), dtype=float)
    all_label_next_errors=np.zeros((N, len_max), dtype=float)
    all_obs_tH_errors=np.zeros((N, len_max), dtype=float)
    all_obs_next_errors=np.zeros((N, len_max), dtype=float)
    counts = np.zeros((len_max), dtype=float)
    
    for label in xrange(Nlabels):
        label_tH_errors[label] = np.zeros((N, len_max),dtype=float)
        label_next_errors[label] = np.zeros((N, len_max),dtype=float)
        obs_tH_errors[label] = np.zeros((N, len_max),dtype=float)
        obs_next_errors[label] = np.zeros((N, len_max),dtype=float)
        
    for t in xrange(N):
        for label in xrange(Nlabels):
            L = results[label]['label_error'][t].shape[0]
            label_tH_errors[label][t,-L:] = results[label]['label_error'][t].mean(axis=1)
            label_next_errors[label][t,-L:] = results[label]['label_error'][t][:,0]
            obs_tH_errors[label][t,-L:] = results[label]['error'][t].mean(axis=1)
            obs_next_errors[label][t,-L:] = results[label]['error'][t][:,0]
        gold_labels = clusters[t][-L:] 
        for i in range(-1,-L-1,-1):
            all_label_tH_errors[t,i] = results[gold_labels[i]]['label_error'][t][i,:].mean()
            all_label_next_errors[t,i] = results[gold_labels[i]]['label_error'][t][i,0]
            all_obs_tH_errors[t,i] = results[gold_labels[i]]['error'][t][i,:].mean()
            all_obs_next_errors[t,i] = results[gold_labels[i]]['error'][t][i,0]
            #stds
            
        counts[-L:] += 1.0
        
    #PLOTS
    #Labels error
    plt.figure()
    plt.subplot('%d%d%d'%(Nlabels+1,1,1))
    plt.title('Label error prediction')
    for j in xrange(len_max):
        plt.scatter(j, all_label_tH_errors[:,j].sum()/float(counts[j]), color=colors2[gold_labels[j]], marker='.', label='tH best ')
        plt.scatter(j, all_label_next_errors[:,j].sum()/float(counts[j]), color=colors2[gold_labels[j]], marker='x', label='next best ')
    plt.ylim((all_label_tH_errors.min()-0.1,all_label_tH_errors.max()+0.1))
    plt.xlim(0, len_max)
    #plt.legend(loc=3,ncol=2)
    for label in xrange(Nlabels):
        plt.subplot('%d%d%d'%(Nlabels+1,1,label+2))
        plt.plot(t_len, label_tH_errors[label].sum(0)/counts, color=colors2[label], linestyle='--', label='tH label %d'%label)
        plt.plot(t_len, label_next_errors[label].sum(0)/counts, color=colors2[label], linestyle='-', label='next label%d'%label)
        plt.ylim((all_label_tH_errors.min()-0.1,all_label_tH_errors.max()+0.1))
        plt.xlim(0, len_max)
        plt.xlim(0, len_max)
        #plt.legend(loc=3,ncol=2)    
    plt.show()
    plt.savefig(filename+'label_errors_avg.jpg')
    
    #Obs error
    #Labels error
    plt.figure(figsize=(20,30))
    plt.subplot('%d%d%d'%(Nlabels+1,1,1))
    plt.title('Label error prediction')
    for j in xrange(len_max):
        plt.scatter(j, all_obs_tH_errors[:,j].sum()/float(counts[j]), color=colors2[gold_labels[j]], marker='.', label='tH best ')
        plt.scatter(j, all_obs_next_errors[:,j].sum()/float(counts[j]), color=colors2[gold_labels[j]], marker='x', label='next best ')
    plt.ylim((all_obs_tH_errors.min()-0.1,all_obs_tH_errors.max()+0.1))
    plt.xlim(0, len_max)
    #plt.legend(loc=3,ncol=2)
    for label in xrange(Nlabels):
        plt.subplot('%d%d%d'%(Nlabels+1,1,label+2))
        plt.plot(t_len, obs_tH_errors[label].sum(0)/counts, color=colors2[label], linestyle='--', label='tH label %d'%label)
        plt.plot(t_len, obs_next_errors[label].sum(0)/counts, color=colors2[label], linestyle='-', label='next label%d'%label)
        plt.ylim((all_obs_tH_errors.min()-0.1,all_obs_tH_errors.max()+0.1))
        plt.xlim(0, len_max)
        plt.xlim(0, len_max)
        plt.legend(loc=1, bbox_to_anchor=(1.3,1.1))
    plt.show()
    plt.savefig(filename+'obs_errors_avg.jpg')
    
    data_stats = {}
    data_stats['label_tH'] = (np.sum(all_label_tH_errors.sum(0)/counts),\
                              np.sqrt(1/(counts-1.0)*np.sum((all_label_tH_errors - all_label_tH_errors.sum(0)/counts)**2)) )
    data_stats['label_next'] = (np.sum(all_label_next_errors.sum(0)/counts),\
                              np.sqrt(1/(counts-1.0)*np.sum((all_label_next_errors - all_label_next_errors.sum(0)/counts)**2)) )
    data_stats['obs_tH'] = (np.sum(all_obs_tH_errors.sum(0)/counts),\
                              np.sqrt(1/(counts-1.0)*np.sum((all_obs_tH_errors - all_obs_tH_errors.sum(0)/counts)**2)) )
    data_stats['obs_next'] = (np.sum(all_obs_next_errors.sum(0)/counts),\
                              np.sqrt(1/(counts-1.0)*np.sum((all_obs_next_errors - all_obs_next_errors.sum(0)/counts)**2)) )
    if save:
        f = open(filename+'all_errors.pkl', 'wb')
        pickle.dump(data_stats,f)
        pickle.dump(all_label_tH_errors, f)
        pickle.dump(all_label_next_errors, f)
        pickle.dump(all_obs_tH_errors, f)
        pickle.dump(all_obs_next_errors, f)
        pickle.dump(label_tH_errors, f)
        pickle.dump(label_next_errors, f)
        pickle.dump(obs_tH_errors, f)
        pickle.dump(obs_next_errors, f)
        f.close()
        
    return


def plot_power_spectrum(data, time_step=1/float(30), ):
    import matplotlib.pyplot as plt
    f = plt.figure()
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], ps.ravel()[idx])
    plt.show()
    f.savefig('results/data_PS.pdf')
    
def plot_data(tfeats, tdata, psr,  pred_2, error_2, pred_1, error_1, tstates, filename):
    size_tr = [len(error_2[i]) for i in xrange(len(error_2))] 
    if error_1 is None:
        error_1 = error_2
        pred_1 = pred_2
    
    N = np.min([200, tdata.obs.shape[1]])
    pred_fut = psr.predict_future(tstates, tdata.fut_act)
    err_fut = np.linalg.norm(tdata.fut_obs - pred_fut)
    pred_obs = psr.predict(tstates, tdata.act)
    err_oo = np.linalg.norm(tdata.obs - pred_obs)
    plt.figure()
    plt.plot(np.arange(N), tdata.obs[0,:N],color = 'r', label='obs')
    plt.plot(np.arange(N), pred_fut[0,:N], color='b', label='pfut')
    plt.plot(np.arange(N), pred_obs[0,:N], color='g', label='pobs')
    [plt.axvline(x=tfeats.locs[i], color='c') for i in xrange(len(tfeats.locs)) if tfeats.locs[i] < N ]
    plt.legend(loc='upper left')
    plt.savefig(filename+'prediction_fut_obs.pdf')
    
    #PLOTTING
    colors=['b', 'g', 'r', 'c', 'k', 'y', 'm','b', 'g', 'r', 'c', 'k', 'y', 'm','b', 'g', 'r', 'c', 'k', 'y', 'm']
    
    f, ax = plt.subplots(len(size_tr), sharex=True, sharey=True)
    if len(size_tr)==1:
        ax=[ax]
    for i in xrange(len(size_tr)):
        ax[i].plot(np.arange(N), error_2[i][:N], color=colors[i])
        ax[i].legend(['%d'%size_tr[i]])
        #ax[i].set_ylim([0, 0.0035])
        ax[i].set_xlim([0, N])
        #ax[i].set_yscale('log')
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    ax[0].set_title('absolute prediction error vs.training size')
    f.savefig(filename+'error_trainingsize.pdf')
    
    f  = plt.figure()
    for i in xrange(len(size_tr)):
        plt.plot(np.arange(N), error_2[i][:N], color=colors[i], label='%d'%size_tr[i])
    #plt.ylim([-0.1,10.0])
    plt.legend()
    f.savefig(filename+'error_trainingsize1.pdf')
    
    
    
    plot_trajs(error_2, batch=size_tr[0], filename=filename)
    
    
    plt.figure()
    plt.boxplot(error_2, 1)
    plt.xticks(np.arange(len(size_tr))+1, size_tr)
    #plt.ylim([-plt.ylim()[1]*0.1, plt.ylim()[1]])
    plt.savefig(filename+'error_minmax.pdf')
    
    f, ax = plt.subplots(len(size_tr), sharex=True, sharey=True)
    if len(size_tr)==1:
        ax=[ax]
    for i in xrange(len(size_tr)):
        ax[i].plot(np.arange(N), (error_2[i])[:N], color=colors[i], label='%da.r.'%size_tr[i])
        ax[i].plot(np.arange(N), (error_1[i])[:N], '--', color=colors[i], label='%db.r.'%size_tr[i])
        ax[i].legend()
        ax[i].set_xlim([0, N+10])
        ax[i].set_yscale('log')
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    ax[0].set_title(' error after refinement - error before')
    f.savefig(filename+'error_refinement_ba.pdf')
   
    
    f, ax = plt.subplots(len(size_tr), sharex=True, sharey=True)
    if len(size_tr)==1:
        ax=[ax]
    for i in xrange(len(size_tr)):
        if pred_1 is not None:
            ax[i].plot(np.arange(N), pred_1[i][0,:N], '--', color=colors[i], label='%d_b.r.'%size_tr[i])
        ax[i].plot(np.arange(N), pred_2[i][0,:N], '-', color=colors[i], label='%d_a.r.'%size_tr[i])
        ax[i].plot(np.arange(N), tdata.obs[0,:N], color='m', label='')  ##ERROR is lens
        ax[i].legend()
        ax[i].set_xlim([0, N])
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    ax[0].set_title('predictions')
    f.savefig(filename+'pred_true_train.pdf')
    
    return  
