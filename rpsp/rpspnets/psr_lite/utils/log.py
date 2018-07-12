#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:55:45 2017

@author: ahefny
"""

import threading
import time
import cPickle
import os
import os.path

class Logger:
    _instance = None
        
    def __init__(self):
        self.global_tag = None
        self.filter = lambda gtag,tag: True  
        self.log = []
        self._file = None
        
        self._lock = threading.Lock()        
        self._period = 5
        self._active = True
        self._save_thread = threading.Thread(target=self._save_loop)
        self._save_thread.start()        
                    
    def append(self, tag, value, print_out=False):
        if self.filter(self.global_tag, tag):
            # If value is a function, call it.
            if callable(value):
                value = value()
                        
            self._lock.acquire()
            self.log.append((self.global_tag, tag, value))
            self._lock.release()
            
            if print_out:
                print 'LOG[{}::{}]:'.format(self.global_tag, tag)
                print value
                    
    def set_file(self, f):  
        assert self._file is None        
        directory = os.path.dirname(f)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        self._file = open(f, 'wb')
                
    #def is_main_thread_alive(self):
    #    for t in threading.enumerate():
    #        if t.name == 'MainThread':
    #            return t.is_alive()
    
    def stop(self):
        self._active = False
        self._save_thread.join()
        self._save()
        
        if self._file is not None:
            self._file.close()
        
    def _save(self):
        self._lock.acquire()
        if self._file is not None:
            for x in self.log:
                cPickle.dump(x, self._file, protocol=2)    
                
            self.log = []
            self._file.flush()

        self._lock.release()
                   
    def _save_loop(self):
        while self._active:
            time.sleep(self._period)
            
            self._save()
                                
    @classmethod
    def instance(cls):
        if Logger._instance is None:
            Logger._instance = Logger()
            
        return Logger._instance
        
if __name__ == '__main__':
    log = Logger.instance()
    test_file = '/tmp/psr_lite_log.pkl' 
    
    log.set_file(test_file)
    log.append('test', 123, print_out=True)   
    log.period = 1
    time.sleep(2)
    log.append('test', 456, print_out=True)       
    log.stop()
    
    f = open(test_file, 'rb')
    
    while True:
        try:
            print cPickle.load(f)
        except EOFError:
            break
        
    f.close()
    
    