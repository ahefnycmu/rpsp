#!/usr/bin/env python2
import git
import os, os.path as osp
from git.exc import GitCommandError
git_ssh_identity_file = os.path.expanduser('~/.ssh/RPSP')
git_ssh_cmd = 'ssh -i %s' % git_ssh_identity_file

import json
from distutils.dir_util import mkpath


class Git_IF(object):
    def __init__(self, path='/mnt/home2/zitamarinho/planning_for_psrs', \
                 git_url='git@bitbucket.org:ahmed_s_hefny/planning_for_psrs.git'):
        self._path = path
        self._head_cmt = None
        self._current_cmt = None
        self._repo = None
        self._git_url = git_url
        #self._best_rwds = 0.0
        #self._best_hex = 'None'
        
    def ping_git(self):
        # rorepo is a Repo instance pointing to the git-python repository.
        # For all you know, the first argument to Repo is a path to the repository
        # you want to work with
        print('ping Repo...')
        self._repo = git.Repo(self._path)
        assert not self._repo.bare
        self._working_dir = self._repo.working_tree_dir
        origin = self._repo.remotes.origin
        #check head version is the same as latest
        assert origin.exists()
        for ref in origin.refs:
            if ref.name=='origin/HEAD':
                self._head_cmt = ref.commit #to get hex cmt.hexsha
                break;
        if self._current_cmt is None:
            return True
        if self._head_cmt.hexsha == self._current_cmt.hexsha:
            return False
        print('done')
        return True     
    
    @property
    def _head_hex(self):
        return str(self._head_cmt.hexsha)[:7]    

    def clone_repo(self, branch='master', rdir=''):
        print('clone Repo...')
        repo_dir = rdir.rstrip('/') + '/cloned_%s/'%(self._head_hex)
        mkpath(repo_dir)
        print repo_dir
        try:
            cloned_repo = git.Repo.clone_from(self._git_url, repo_dir, branch=branch)
        except GitCommandError:
            cloned_repo = git.Repo(repo_dir)
            assert not self._repo.bare
        print('done')
        return cloned_repo
    
    def del_repo(self, cloned_repo ):
        print('Deleting Repo...')
        from shutil import rmtree
        print 'remove cloned repo: %s'%(cloned_repo.working_dir)
        rmtree(cloned_repo.working_dir)
        print('done')
        return
        
    def tag_git(self, cloned_repo, tag='tL.id.T', cmt=''):
        '''
        tag git repo new commit after validating
        provide tag name lightweight: tL.id or tF.id
        and commit hash cmt
        '''
        new_tag = cloned_repo.create_tag(tag, message="Validate commit %s with test %s" %(cmt,tag))
        cloned_repo.remotes.origin.push(new_tag)
        print 'TAG is ', tag, cmt
        return
    

    