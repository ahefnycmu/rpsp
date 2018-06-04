#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:38:44 2017

@author: zmarinho, ahefny
"""
import numpy as np
import numpy.linalg as npla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import gmtime,strftime
import operator
from distutils.dir_util import mkpath
from _collections import defaultdict
import cPickle as pickle
import re
import glob,os
matplotlib.rcParams.update({'font.size': 20})
#matplotlib.rcParams.update({'figure.figsize': (6.5,6.5)})
matplotlib.rcParams.update({'figure.dpi': 300})
matplotlib.rcParams.update({'xtick.labelsize': 15})
matplotlib.rcParams.update({'ytick.labelsize': 15})
#matplotlib.rcParams.update({'figure.autolayout':True})

def dfn_key(orig_key):
    fn = lambda key: defaultdict(lambda:orig_key,{})[key]
    return fn

class call_plot(object):
    def __init__(self, n=3, sharex=True, sharey=False, name=strftime('results/%d_%b_%H.%M.%S/', gmtime()), trial=0):
        self.fig, self.ax = plt.subplots(n, sharex=sharex, sharey=sharey)
        self.iter=0
        self.trial = trial
        self.path = '/'.join(name.rsplit('/')[:-1])+'/'
        mkpath(self.path)
        self.name = name

    def plot(self, cost_m, cost_s, error_m, error_s, rwd_m, rwd_s, change_clr=False,
            label_1='pred_error', label_2 ='grad_ctg', label_3='reward' ):
        marker = 'o'
        if change_clr:
            marker='^'
        self.scatter(self.ax[0], cost_m, cost_s, 'k', marker=marker)
        self.ax[0].set_ylabel(label_1)
        self.scatter(self.ax[1], error_m, error_s, 'r',marker=marker)
        self.ax[1].set_ylabel(label_2)
        self.scatter(self.ax[2], rwd_m, rwd_s, 'b', marker=marker)
        self.ax[2].set_ylabel(label_3)
        
        self.iter+=1
        self.fig.savefig(self.path+'error_T%d.pdf'%self.trial)
       
    def plot_single(self, rwd_m, rwd_s ):
        self.scatter(self.ax, rwd_m, rwd_s, 'b')
        self.ax.set_ylabel('reward')
        self.iter+=1
        self.fig.savefig(self.path+'error_T%d.pdf'%self.trial)
        
    def scatter(self, ax, value, std, color, marker='o'):
        ax.errorbar(self.iter, value, yerr=std, fmt=marker, color=color, ecolor='k', capthick=2)
        return
      
    def plot_traj(self, t, pred, name=''):
        N = t.obs.shape[1]
        f,ax = plt.subplots(N, sharex=True, sharey=False)
        if N == 1:
            ax = [ax]
        err = np.linalg.norm(t.obs-pred, axis=1)
        ts = np.arange(len(err))
        for i in xrange(N):
            ax[i].plot(ts, pred[:,i], '-', color='r')
            ax[i].plot(ts, t.obs[:,i], 'o', color='k')
            ax[i].set_ylabel('pred%d'%self.iter)
        f.savefig(self.path+'traj0_T%d_iter%d_%s.pdf'%(self.trial, self.iter, name))
        return
    
    
def load_model(fname, fdir):
    f = open(fdir+fname, 'rb')
    model_results = pickle.load(f)
    save_args = pickle.load(f)
    f.close()
    if isinstance(model_results, dict):
        print_short_description_results([model_results], save_args)
    else:
        print_short_description_results(model_results, save_args)
    return model_results, save_args

def load_params(fname, fdir=''):
    if fdir<>'':
        fdir = '/'.join(fdir.split('/')[:-1])+'/'
    try:
        f = open(fdir+fname, 'rb')
        params = pickle.load(f)
        f.close()
        print('Loading params: ', params.keys())
    except IOError:
        params={}
        print('Non existing pickle file', fdir+fname)
    return params

def save_params(params, fname, fdir=''):
    try:
        f = open(fdir+fname, 'wb')
        pickle.dump(params, f)
        f.close()
        print('Saving params: ', params.keys())
    except IOError:
        print('Error saving file')
        embed()
    return 

def print_short_description_results(results, args):
    best = [];    last = [];    mean = [];    std = [];
    print 'results description:'
    for trial in xrange(len(results)):
        best.append( np.max( [np.mean(results[trial]['rewards'][it]) for it in xrange(len(results[trial]['rewards'])) ] ))
        argbest = np.argmax( [np.mean(results[trial]['rewards'][it]) for it in xrange(len(results[trial]['rewards'])) ] )  
        mean.append( np.mean(results[trial]['rewards'][-1]) )
        std.append( np.std(results[trial]['rewards'][-1]) )
        print 'trial %d: best[iter %d]=%.3f, last[%d iter] mean[len %d]=%.3f +-%.3f  '%(trial, argbest, best[-1], len(results[trial]['rewards']),\
                                                                             len(results[trial]['rewards'][0]), mean[-1], std[-1]) 
        results[trial]['best_rwd'] = best 
        results[trial]['mean_rwd'] = mean
        results[trial]['std_rwd'] = std
        results[trial]['last_rwd'] = last
    return
        
def save_model( fname, fdir, model_results,args):
    f = open(fdir+fname+'.pkl', 'wb')
    try:
        pickle.dump(model_results, f, protocol=2)
        pickle.dump(args, f)
    except IOError:
        print 'Saving error model results'
        embed()
    f.close()
    return 



def plot_trials(results, fdir, f=None, shape='-s',name='rewardstrials',fig=None,step=1):
    '''plot mean over iterations'''
    if f is None:
        f = lambda t: results[t]['rewards']
    lenf = len(results)
    if fig is None:
        fig = plt.figure()
    ctgs=[[] for i in xrange(len(f(0)))]
    for t in range(lenf):
        for it in xrange(0,len(f(t)),1):
            ctgs[it].extend(f(t)[it])
    m = [np.mean(ctgs[it]) for it in xrange(0,len(ctgs),step)]
    stds = [np.std(ctgs[it])/3 for it in xrange(0,len(ctgs),step)]
    kwargs={'label':name,
                'yerr':stds, 
                'fmt':shape,
                'capsize':0,
                'ms':10,#8
                'elinewidth':2}
    plt.errorbar(np.arange(0,len(ctgs),step),m, **kwargs)
    plt.legend(loc='upper center', bbox_to_anchor=(0.55, 1.15), 
          frameon=False, ncol=4, columnspacing=0.2, handletextpad=0.0, fontsize=18) #ncol=6, fontsize=18
    plt.xlabel('experience ($10^4$ timesteps)')
    fig.savefig(fdir+name+'.png')
    return fig

def get_logs(fdir, fname, string='Wpred'):
    f=open(fdir+fname, 'r')
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)    
    out =[];
    for line in f.readlines():
        if line.find(string)<>-1:
            values = rx.findall(line)
            data = np.asarray(values, dtype=float)
            out.append(data)
    f.close()
    return out

def plot_rows(data, labels=[], path='', ylim=None):
    ''' data has a subplot for each column and '''
    data = np.asarray(data)
    if data.ndim==1:
        data = data.reshape(1,-1)
    N,ndims = data.shape
    ts = np.arange(N)
    cmap = plt.cm.get_cmap('spectral',ndims)
    f,ax = plt.subplots(ndims, sharex=True, sharey=False)
    if ndims==1:
        ax=[ax]
    for i in xrange(ndims):
        ax[i].plot(ts, data[:,i], '-', color=cmap(i), label='%d'%i if len(labels)==0 else labels[i])
        ax[i].legend(loc=9)
        if ylim is not None:
            ax[i].set_ylim(ylim[0],ylim[1])
    f.savefig(path+'data.pdf')
    return
  
def mean_lists(Xlists):
    max_len = np.max([len(t) for t in Xlists])
    mean_=np.zeros((max_len,len(Xlists[0][0])))   
    counts_ = np.zeros((max_len))   
    for i in xrange(max_len):
        for t in Xlists:
            if len(t)>i:
                mean_[i]+=np.asarray(t[i])
                counts_[i]+=np.asarray(1.0)
        mean_[i]/=float(counts_[i])  
    return mean_

def get_log_key(fdir, string='', plot=False):
    import re
    import glob
    total_data = []
    for fname in glob.glob(fdir+'log_*'):
        print (fname)
        data = get_logs('', fname, string=string)
        total_data.append(data)
        if plot:
            plot_rows(data, path=fname+string, labels=[])
    return total_data

def plot_from_logs(fdir):
    import re
    import glob
    total_vars = []
    total_mag = []
    labels=[]
    for fname in glob.glob(fdir+'log_*'):
        print (fname)
        data_pred = get_logs('', fname, string='Wpred=')
        data_fm = get_logs('', fname, string='Wfm=')
        if len(data_pred)==0:
            continue
        if len(data_fm)==0:
            vars = list(zip(*[zip(*data_pred)[0], zip(*data_pred)[1]]))
            labels=['pred_var','pred_mag']
            total_vars.append(vars)
        else:
            vars = list(zip(*[zip(*data_pred)[0], zip(*data_pred)[1]]))
            labels=['pred','fm']
            total_vars.append(vars)
            if len(zip(*data_pred))>1:
                magnitude = list( zip(*[zip(*data_pred)[1], zip(*data_fm)[1] ] ) )
                total_mag.append(magnitude)
                plot_rows(magnitude,path=fname+'_mag_', labels=labels)
        plot_rows(vars,path=fname+'_vars_', labels=labels)
    if len(total_vars)==0:
        print('No logs for weight evolution!')
        return
    mean_vars = mean_lists(total_vars)
    #mean_vars = np.mean(total_vars,axis=0)
    plot_rows(mean_vars,path=fdir+'total_vars_', labels=labels)#, ylim=[0,15])
    if len(total_mag)>0:
        #mean_mag = np.mean(total_mag, axis=0)
        mean_mag = mean_lists(total_mag)
        plot_rows(mean_mag,path=fdir+'total_mag_', labels=labels)


def load_experiment(fdir, filename, best=1, max_trials=5, save_csv=True):
    ''' load a saved model and get the rewards 
    Example:  python stats_test.py '''
    from operator import itemgetter
    try:
        model_results, args = load_model(filename, fdir) 
    except IOError:
        print fdir, filename
        name = filename.split('/')
        filedir = '/'.join(fdir.rstrip('/').split('/')+name[:-1])+'/'
        model_results = []
        args = None
        for filename in glob.glob(filedir+"*.pkl"):
            print 'filename', filename
            try:
                results, arg = load_model(filename, '')
            except Exception:
                embed()
            args = arg
            model_results.append(results)
            args.repeat = len(model_results)
    if len(model_results)==0:
        model_results=None 
        args={}
        Rbest = np.zeros((5000))
        counts= np.ones((5000))
        return Rbest,counts, model_results, args
    trials = len(model_results)
    T = args.iter
    MAX = np.min([best,len(model_results)])
    print 'using best ', MAX,' results out of %d! '%max_trials, filename
    R = []; counts=[]; rwds_mean=[];
    num_iter = 0
    for t,trial in enumerate(model_results[:max_trials]):
        R.append( np.zeros((T), dtype=float) )
        counts.append( np.zeros((T), dtype=float) )
        rwds_mean.append( np.zeros((T), dtype=float) )
        for i,rwds in enumerate(trial['rewards']):
            R[t][i] += np.sum(rwds)
            counts[t][i] += len(rwds)
            rwds_mean[t][i] += np.mean(rwds)
            
    R = np.asarray(R)
    counts = np.asarray(counts)
    best_trials = sorted([(i,np.sum(r[:])) for i,r in enumerate(rwds_mean)], key=itemgetter(1),reverse=True)
    best_trials = zip(*best_trials)[0][:MAX]
    print 'best', best_trials
    N_per_iter = counts[list(best_trials),:].T
    R_per_iter = R[list(best_trials),:].T

    if save_csv:
        import csv
        print('Saving csv', filedir)
        filecsv = open(filedir+'results_%d.csv'%len(R_per_iter.T), 'w')
        cwriter = csv.writer(filecsv, delimiter=',')
        R_per_traj = R_per_iter/(N_per_iter+1.0e-12)
        for n in xrange(len(R_per_iter.T)):
            cwriter.writerow(R_per_traj[:,n])
        filecsv.close()
    return R_per_iter, N_per_iter, model_results, args
    

def process_results(R,c, offset=0):
    num_iter = R.shape[0]
    N = len(c[0]) if isinstance(c[0], np.ndarray) else 10.0
    R_per_iter = [ [(R[i]/(c[i]+1e-12)).mean() for i in xrange(num_iter)],[(R[i]/(c[i]+1e-12)).std()/np.sqrt(N) for i in xrange(num_iter)]]
    Reward = [R[i]/(c[i]+1e-12) for i in xrange(num_iter)]
    R_cum_iter = [ np.cumsum(Reward,0), np.cumsum(Reward,0)*0.0]
    AUC = np.sum(Reward,0)
    R_per_iter[0] = [0.0]*offset + R_per_iter[0]
    R_per_iter[1] = [0.0]*offset + R_per_iter[1]
    return R_per_iter, R_cum_iter, AUC

def config_plot(ax, xlabel,ylabel,ylim, title=None):
    
    if isinstance(ax, matplotlib.axes.Subplot):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim is None:
            ax.set_ylim([-10,plt.ylim()[1]])
        else:
            ax.set_ylim(ylim)
    else:
        #plt.title(title)
        ax.xlabel(xlabel)
        ax.ylabel(ylabel)
        if ylim is None:
            ax.ylim([-10,plt.ylim()[1]])
        else:
            ax.ylim(ylim)
        #plt.title(title)=
        #plt.gca().set_aspect(9)
        #plt.subplots_adjust(bottom=0.25)
    return

def plot_models(Y, names=None, fdir='', step=1, length=np.inf,\
                figname='rwds_per_iter', title='', ylabel='', \
                xlabel='experience ($10^3$ timesteps)', \
                cdict=lambda k: 'k',timestep=1,
                shape_dict=dfn_key('--.'),stds=1.0,\
                ioff=0, ncol=4, ylim=None,f=None, ax=None, \
                legend=True, kfold=10.0**4, **kwargs_plot):
    ''' plot for all models in Y = [(means, stds)...(means,stds)]'''
    min_len_res = np.min([len(Y[i][0]) for i in xrange(len(Y)) ])
    min_len = np.min([min_len_res,length])
    print 'using episode length ', min_len
    
    t = np.arange(0,min_len*timestep/float(kfold)*10.0,timestep/float(kfold)*10.0)
    if f is None:
        f = plt.figure(frameon=False)
        ax = plt
    for i,y in enumerate(Y):
        error = [e/stds for e in y[1][:min_len-ioff:step]]
        kwargs={'label':names[i],
                'yerr':error, 
                'fmt':shape_dict(names[i]),
                'mfc':cdict(names[i]),
                'capsize':0,
                'ms':10,#8
                'elinewidth':2}
        kwargs.update(kwargs_plot)
        try:
            kwargs['color']=cdict(names[i])
            kwargs['mfc']=cdict(names[i])
            kwargs['mec']=cdict(names[i])
        except Exception:
            pass
        ax.errorbar(t[ioff::step], y[0][:min_len-ioff:step], \
                     **kwargs)
    if legend:
        legend=ax.legend(loc='upper center', bbox_to_anchor=(0.55, 1.15), 
          frameon=False, ncol=ncol, columnspacing=0.2, handletextpad=0.0, fontsize=18) #ncol=6, fontsize=18
        legend.get_frame().set_facecolor('white')
        config_plot(ax, xlabel, ylabel, ylim, title)
    f.savefig(fdir+figname+'.png',bbox_inches='tight',dpi=300, pad_inches=0.1)
    return

def plot_rwds( labels, cdict=(lambda k: 'k'), fdir='', model_paths=[], \
              BEST=3, MAX_LEN=300, MAX_TRIALS=5, STEP=10, title='',step=1,\
              shape_dict=dfn_key('--o'), ncol=4, ylim=None, f=None, ax=None, legend=True, kfold=10**3):
    R_iter = []; R_cum =[]; all_AUC=[];
    #TODO plot MSE for LSTMs and PSRnet
    print 'showing %d models: '%len(labels), labels
    for m in xrange(len(labels)):
        model_name = labels[m]
        if model_name[:3]=='GRU' or model_name[:3]=='PSD':
            R_per_iter, R_cum_iter, AUC = csv2R_convert_results(fdir, model_paths(model_name), shape=MAX_LEN)
            
            
        else:
            R,c,model_results,args = load_experiment(fdir, model_paths(model_name), best=BEST, \
                                                     max_trials=MAX_TRIALS, save_csv=True)
            offset=0
            if (labels[m][:4]=='RPSP' or labels[m][-4:]=='+obs'):
                offset = 10
                if title.split()[0]=='Swimmer':
                    offset=5
            
            R_per_iter, R_cum_iter, AUC = process_results(R,c, offset=offset)
        R_iter.append(R_per_iter)
        R_cum.append(R_cum_iter)
        all_AUC.append(AUC)
    plot_models(R_iter, names=labels, fdir=fdir+' '.join(title.split()[:-1]), step=STEP, length=MAX_LEN,\
                figname='_rwdperiter_best%d_step%d'%(BEST,STEP), \
                title=title, \
                ylabel=title,
                xlabel='experience ($%d$ timesteps)'%kfold, 
                cdict=cdict,timestep=step,\
                shape_dict=shape_dict, ncol=ncol, ylim=ylim, f=f, ax=ax, legend=legend, kfold=kfold)
    #plot_models(R_cum, names=labels, fdir=fdir+title, step=STEP, length=MAX_LEN,\
    #            figname='_cumulrwd_best%d_step%d'%(BEST,STEP), \
    #            title=title, \
    #            ylabel='Cummulative returns (%d trajectories, %d trials)'%(len(R_per_iter), len(R_per_iter[0])),
    #            cdict=cdict)
    print 'done'
    return R_iter, R_cum, all_AUC

def csv2R_convert_results(fdir, path, shape=0):
    import csv
    R_per_iter = []
    R_cum_iter = [[0,0]]
    try:
        freader = csv.reader(file(fdir+path), delimiter=',')
        #avg = np.mean(trial_means, axis=0)      
        #std = np.std(trial_means, axis=0)
        #cumulative = np.sum(trial_means,axis=1)
        #results = np.stack([avg, std]).T
        results = []
        for line in freader:
            results.append(line)
            #rmean = float(line[0])
            #rstd = float(line[1])
            #R_per_iter.append([rmean, rstd])
            #R_cum_iter.append([R_cum_iter[-1][0]+rmean , rstd])
        results = np.asarray(results, dtype=float)
        print 'LEN ', results[:,0].shape
        R_per_iter = [[np.mean(r),np.std(r)/np.sqrt(float(len(r)))] for r in results.T]
        R_cum = np.cumsum(results, axis=1)
        R_cum_iter = [[np.mean(R_cum[:,r]),np.std(R_cum[:,r])] for r in xrange(results.shape[1])] 
        AUC = np.sum(results, axis=1)
    except IOError:
        print ('load error', fdir, path)
        embed()
        R_per_iter = [[0,0]*shape]
        R_cum_iter = [[0,0]*shape]
    return zip(*R_per_iter), zip(*R_cum_iter), AUC



def compute_AUC(R_cum):
    ''' R_cum: label x 2(mean,std) x num_iter'''
    mAUC=[]; sAUC=[]
    for i in xrange(len(R_cum)):
        mAUC.append(np.mean(R_cum[i][0][-1]))
        sAUC.append(np.std(R_cum[i][0][-1]))
    return mAUC, sAUC


def autolabel(rects, idxs, value_increment=1000):
    """
    Attach a text label above each bar displaying its height
    """
    for c in range(len(rects)):
        prev_height=np.inf
        for r in range(len(rects[c])):
            rect = rects[c][idxs[r,c]].patches[0]
            height = rect.get_height()
            #print height, prev_height-5*value_increment
            plt.text(rect.get_x() +  rect.get_width()+0.01, np.min([height,prev_height-rect.get_axes().get_ylim()[1]/20]),#-13
                    '%d' % int(np.round(height,decimals=-3)/value_increment),
                    ha='left', va='top',fontsize=12)
            prev_height = np.min([height, prev_height])

def plot_table(R_cum, clabels, rlabels, title, ylabel, fdir, plot_table=False, incr=1000,\
               ncol=4):
    ''' plot table results 
    R_cum: receive cumulative reward
    clabels has all rlabels for each column
    data: columns as categories rows as types
    '''
    
    mAUC, sAUC = compute_AUC(R_cum)
    columns = clabels
    rows = rlabels
    value_increment = incr
    maxval = np.round(np.ceil(np.max(mAUC)/value_increment),decimals=-3)
    values = np.arange(0, maxval, 50)
    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(rlabels)
    n_cols = len(clabels)
    data = np.asarray(mAUC).reshape(n_cols,n_rows).T #cols as categories in clabels and rows as types in rlabels
    sdata = np.asarray(sAUC).reshape(n_cols,n_rows).T
    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4
    index_sorted = [np.asarray(sorted(zip(data[:,c], np.arange(n_rows)),key=operator.itemgetter(0), reverse=True))[:,1] for c in range(n_cols)]
    index_sorted = np.asarray((zip(*index_sorted)),dtype=int)
    
    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.array([0.0] * len(columns))
    # Plot bars and create text labels for the table
    cell_text = []; rects=[[[]]*n_rows]*n_cols
    pos = np.linspace(0, bar_width, n_rows+2)
    f = plt.figure(frameon=False)
    for row in range(n_rows):#-1,-1,-1):
        for c in range(n_cols):
            idx = index_sorted[row,c]
            rect = plt.bar(index[c], data[idx,c], bar_width, bottom=y_offset[c], color=colors[idx], edgecolor = "none")
            plt.errorbar(index[c] + pos[row+1], data[idx,c], sdata[idx,c], color='k')#lw=2, capsize=5, capthick=2,
            rects[c][idx]=rect
        #y_offset = y_offset + data[row]
        if plot_table:
            cell_text.append(['%1.1f' % (x/float(value_increment)) for x in y_offset])
    
    if plot_table:
        # Reverse colors and text labels to display the last value at the top.
        colors = colors[::-1]
        cell_text.reverse()
        
        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              rowColours=colors,
                              colLabels=columns,
                              loc='bottom')
    
        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2)
    
    plt.ylabel(ylabel)
    #plt.yticks(values * value_increment, ['%d' % val for val in values])
    #plt.set_yticklabels(x, minor=False)
    autolabel(rects, index_sorted, value_increment)       
    if not plot_table:
        plt.xticks(np.arange(n_cols)+0.5, clabels)
    plt.title(title)
    plt.legend(rects[0], rlabels, loc='upper center',  bbox_to_anchor=(0.5, 1.0),#0.5,1.15,
               frameon=False, ncol=ncol, columnspacing=0.2, handletextpad=0.0, fontsize=18)
    #plt.show()
    plt.box('off')
    plt.tick_params(top='off', bottom='on', left='on', right='off', labelleft='on', labelbottom='on')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    f.savefig(fdir+title+'.png',bbox_inches='tight',dpi=300, pad_inches=0.1)
    return
    




if __name__ == '__main__': 
    pass