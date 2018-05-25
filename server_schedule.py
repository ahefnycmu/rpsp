#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''

A server schedule script to check for git updates and perform a sanity test on the latest commit. 
saves data to a log file. Runs with lite frequency a lite test and full test with fullfrequency. 
If no new commits have been done NoOp. 
Run :

python server_schedule.py server pwd_file logfile lite_checkfrequency full_checkfrequency
'''
import numpy as np 
import argparse
from pytz import utc
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
import json
from psr_lite.psr_lite.utils.log import Logger
from test_utils.git_utils import Git_IF
from distutils.dir_util import mkpath
import glob, os
import globalconfig
from IPython import embed
import time
import numpy as np
from test_utils.plot import load_model
import subprocess
TOL_RWD={'Swimmer-v1':80 ,'Walker2d-v1':800, 'Hopper-v1':800, 'CartPole-v1': 200}
TOL_CS={'Swimmer-v1':10000 ,'Walker2d-v1':40000, 'Hopper-v1':40000, 'CartPole-v1': 15000}


def load_cfg(fpath):
    f = open(fpath,'r')
    data = json.loads(f.read())
    f.close()
    return data

def rwd2stats(fdir, cmt):
    results, args = load_model('lite-cont.pkl', fdir)
    all_results = [np.mean(results[trial]['rewards'],axis=1) for trial in xrange(len(results))]
    R10max = np.max([r[-10:] for r in all_results])
    R10trial = np.argmax([r[-10:] for r in all_results])
    CSmax = np.max([np.cumsum(r,axis=0) for r in all_results])
    CStrial = np.argmax([np.cumsum(r,axis=0) for r in all_results])
    
    setattr(args, 'CS', (CSmax, CStrial, int(CSmax>=TOL_CS[args.env])))
    setattr(args, 'R10', (R10max, R10trial, int(R10max>=TOL_RWD[args.env])))
    setattr(args, 'cmt', cmt)
    print 'last rewards: ', R10max, 'cumulative: ',CSmax
    #TODO: add best 3/10 avg
    return results, args

def save_stats(fdir, cmt):
    results, args = rwd2stats(fdir,cmt)
    delattr(args, 'rng')
    json.dump(args.__dict__,open(fdir+'/params_ref','w'))
    return


def del_files( fdir, match=''):
    os.chdir(fdir)
    for file in glob.glob(match):
        print('removing',file)
        os.remove(fdir+file)

def execute(cmd,cwd,fout):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,cwd=cwd, shell=True,bufsize=1,universal_newlines=True)
    output=''
    for line in iter(p.stdout.readline, ""):
        fout.write(line)
        output += line
    p.wait()
    exitCode = p.returncode
    if (exitCode == 0):
        return output
    else:
        raise Exception(cmd, exitCode, output)


class TestBed(object):
    ''' refdir contains folders for each environment. Where inside must be a configuration file: params
        filen: contains a folder with the saved configuration
    '''
    def __init__(self, filen, fname, refdir):
        self._envs = ['CartPole-v1', 'Walker2d-v1', 'Hopper-v1', 'Swimmer-v1']
        self._filen=filen
        self._ref_dir= refdir
        self._config = fname
        self.load_data(fname)
        return
    
    def load_data(self, fname):
        config_path = self._ref_dir+fname
        self._config = fname
        self._test_data = load_cfg(config_path)
        env = self._test_data['env']
        assert env in self._envs, 'unknown environment!'
        self._test_env = self._envs.index(env)
        return self._test_data

    @property
    def next_env(self):
        self._test_env = (self.id+1)%len(self._envs)
        env = self.current_env
        return  self.load_data(env+'/params')
    
    @property
    def current_env(self):
        return self._envs[self._test_env]
    @property
    def data(self):
        return self._test_data
    
    @property
    def id(self):
        return np.copy(self._test_env)
    @property
    def env(self):
        return str(np.copy(self._test_data['env']))
    
    @property
    def dump_data(self, data={}, ):
        delattr(args, 'rng')  
        json.dump(data,open(self._report_dir+'/params','w'))
        return
    
    def clean_files(self, fdir):
        match='*.pkl '
        del_files(fdir, match)
           
        
    def test(self, cloned_repo, cmt, validate_file, log_file='log.log'):
        print('Test Repo')
        self._report_dir = self._filen+str(cmt)+'/'+self.env+'/' # can keep this for later if needed
        mkpath(self._report_dir)
        logf = self._report_dir+log_file
        log = open(logf,'a')
        print('dumping to ',self._report_dir+log_file)
        #cmd = ['python', 'call_test.py', '--config', validate_file, '--repeat', trials, '--tfile', test_file]
        #p = subprocess.call(cmd,shell=True,cwd=cloned_repo.working_dir+'/code/',stdout=)
        cmd= 'python -u call_test.py --config %s --tfile %s '%(validate_file,self._report_dir)
        try:
            execute(cmd,cloned_repo.working_dir+'/code',log)
        except Exception:
            print 'Failed running script!'
            embed()
        results, run_args = rwd2stats(self._report_dir, cmt)
        print('done')
        return results, run_args
    
    
    #TODO: validate after getting reference
    def validate_test(self, gif, args, cloned_repo, delete=True):
        print('Validade...')
        #TODO: add best 3/10 avg
        best=[0,0]
        if bool(args.CS[-1]):
            best[0]=1
        if bool(args.R10[-1]):
            best[1]=1
        success=(np.sum(best)==2)
            
        f = open(args.working_dir+'/history_log.log','a')
        final_tag = 'BEST:'+args.env+':'+args.ref_cmt+':CS:'+\
                    args.ref_CS+':R10:'+args.ref_R10+'\n\t'+\
                    'COMMIT:'+args.cmt+':CS:'+args.CS+':R10:'+args.R10
        
        if success:
            #tag success and delete cloned repo and pickles
            final_tag += ':S'
            if delete:
                self._clean_files(self._report_dir)
        else:
            #tag failed keep repo
            final_tag += ':F'
        gif.del_repo(cloned_repo)
        f.write(time.strftime("%d %b %Y %H-%M-%S ---", time.gmtime())+final_tag+'\n')
        f.close()
        print('done')
        return 
    
    def run(self, gif):
        '''
        ping git repo for new commit and run test_script: scripts/run_validate.sh 
        provide tag name lightweight:tag-> tL.id or tF.id.Y  ex:testlightweight.id.S or testfull.id.Fail
        and commit hash: cmt
        '''
        new_commit = gif.ping_git()
        if new_commit:
            print('Checking new_commit!')
            cloned_repo = gif.clone_repo(rdir=self._filen, branch='master')
            tag = gif._head_hex
            print tag
            results, run_args = self.test(cloned_repo, gif._head_hex, self._ref_dir+self._config)
            self.validate_test(results, run_args, cloned_repo)
            self.dump_data(run_args.__dict__())
            return results, run_args
        print('No new commits!')
        return
    
    

#################################
#################################
##           NOT USED :TODO:server schedule
#################################
#################################

jobstores = {
    'default': MemoryJobStore()
}
executors = {
    'default': ThreadPoolExecutor(20),
    'processpool': ProcessPoolExecutor(5)
}
job_defaults = {
    'coalesce': False,
    'max_instances': 2
}
scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults)
lite_jobs = [(0,None)] #id, commit version
full_jobs = [(0,None)] #id,commit version

def myfunc():
    print'runningSERVER'
    return

def _run_server(args):
    #TODO: implement and test scheduler jobs
    validate_config = args.path+args.test_script
    path = args.path
    url = args.git
    log_file = args.log+'.log'
    g = Git_IF(git_url=url,path=path)
    run_lite = lambda: g.run(validate_config, args.working_dir, log_file=log_file, trials=args.lite_freq)
    run_full = lambda: g.run(validate_config, args.working_dir, log_file=log_file, trials=args.full_freq)
    
    lite_sched = scheduler.add_job(run_lite, 'interval', days=args.lite_freq, id='%d_%s'%lite_jobs[-1])
    full_sched = scheduler.add_job(run_full, 'interval', days=args.full_freq, id='%d_%s'%full_jobs[-1])
    return 

#TODO
def test_reference(args, TB):
    validate_config = args.path+args.test_script
    path = args.path
    url = args.git
    log_file = args.log+'.log'
    g = Git_IF(git_url=url,path=path)
    run_full = lambda: g.run(validate_config, args.working_dir, log_file=log_file, trials=args.full_freq)
    return 


def _kill_server(): 
    jobs = scheduler.get_jobs()
    for job in jobs:
        job.remove()
    return


#################################
#################################


def test_lite(args, validate_config='lite_results.json'):
    testbed=TestBed(args.lite_dir, validate_config, args.reference_dir)
    test_params = testbed.next_env
    print('Testing with ',test_params['env'])
    path = args.path
    url = args.git
    g = Git_IF(git_url=url,path=path)
    results, run_args = testbed.run(g)
    embed()    
    return
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--git', type=str, default='git@bitbucket.org:ahmed_s_hefny/planning_for_psrs.git')
    parser.add_argument('--lite_freq', type=int, default=1,help='lite test frequency in days')
    parser.add_argument('--full_freq', type=int, default=7, help='full test frequency in days')
    parser.add_argument('--path', type=str, default='/mnt/home2/zitamarinho/planning_for_psrs', help='git path')
    parser.add_argument('--i', type=str, default='./ssh/RPSP', help='ssh private key')
    parser.add_argument('--test_cfg', type=str, default='/code/scripts/validate.json', help='validation parameters')
    parser.add_argument('--kill',default=False,action='store_true')
    parser.add_argument('--ref',default=False,action='store_true',help='Run full reference tests. Otherwise run lite version.')
    args = parser.parse_args()
    working_dir = args.path.rstrip('/')+'/log_RPSP/'
    mkpath(working_dir)
    print working_dir
    setattr(args,'working_dir', working_dir)
    setattr(args,'reference_dir', working_dir+'reference/')
    setattr(args,'lite_dir', working_dir+'lite/')
    
    
    
    
    
    if args.kill:
        print 'killing servers?'
        embed()
        _kill_server()
    else:
        test_lite(args, validate_config=args.test_cfg)
            
        #_run_server(args)
        
    
