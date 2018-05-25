#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 01:17:32 2017

@author: ahefny
"""

import struct
import fcntl
import os

MAX_BUTTONS = 20
MAX_AXES = 20
EVENT_FORMAT = 'IhBB'
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
MAX_SHORT = 2.0 ** 15

class InputDevice:
    '''
    Provides an interface for polling an input device with buttons and analog inputs. 
    It is possible to have multiple interfaces to the same physical device.
    '''
    def __init__(self, dev_name = '/dev/input/js0', num_buttons=MAX_BUTTONS, num_axis=MAX_AXES):
        self._dev_name = dev_name
        self._btn = [False] * num_buttons
        self._btn_pressed = [False] * num_buttons
        self._btn_released = [False] * num_buttons
        self._axes = [0] * num_axis                
        self._is_opened = False
      
    def copy(self):
        '''
        Creates a new interface to the same device with the same specs.
        '''
        return InputDevice(self._dev_name, len(self._btn), len(self._axes))
        
    def open(self):        
        self._file = open(self._dev_name, 'rb')
        fd = self._file.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)                
        self._is_opened = True
                
    def poll(self):
        if not self._is_opened:
            raise Exception('Cannot poll a closed input device.')
        
        for i in xrange(len(self._btn_pressed)):
            self._btn_pressed[i] = False
            self._btn_released[i] = False
        
        try:
            event = self._file.read(EVENT_SIZE)
                        
            while event:                
                data = struct.unpack(EVENT_FORMAT, event)                
                time, value, type, num = data

                if not (type & 0x80): # Ignore initial events                     
                    if (type & 1):
                        # Button
                        self._btn[num] = bool(value)                        
                        if value:
                            self._btn_pressed[num] = True
                        else:
                            self._btn_pressed[num] = False                                                
                    else:
                        # Axis
                        self._axes[num] = value / MAX_SHORT                                              
                
                event = self._file.read(EVENT_SIZE)
        except Exception as x:
            pass
        
    def get_button(self, index):
        return self._btn[index]
    
    def get_axis(self, index):
        return self._axes[index]
    
    def is_button_pressed(self, index):
        '''
        Returns True of the button specified by index is pressed between the last
        two polls.
        '''
        return self._btn_pressed[index]
    
    def is_button_released(self, index):
        return self._btn_pressed[index]
                                    
    def close(self):
        self._file.close()
        self._is_opened = False

if __name__ == '__main__':
    dev1 = InputDevice('/dev/input/js0')
    dev1.open()
    dev2 = dev1.copy()
    dev2.open()

    while True:
        dev1.poll()           
        dev2.poll()        
                       
        for i in xrange(5):
            if dev1.is_button_pressed(i):
                print 'Button %d pressed on dev1' % i
            
            if dev2.is_button_released(i):
                print 'Button %d released on dev2' % i
                
            #print (i,dev1.get_button(i),dev2.get_axis(i)),

        #print ''            
        #'''