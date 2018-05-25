#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Copyright 2017 [Ahmed Hefny]
'''

'''
A utility for sanity checking whether two programs produce identical numerical outputs.

The checked programs act as clients to the debug server.
Program executation is divided into check-point. In check point, the clients 
send name/value pairs (string/numpy array) to the server and the server ensures
they are identical otherwise both clients start the debugger. In this case, 
Checkpoint data can be examined through local_data variable. Data sent by the other program
are stored in remote_data.

The server is started by running this script. Clients interact with the server
through the methods connect_to_server, send_data and check_point.

Checkpoint pass criteria can be changed by reimplementing _check_data function.
The curretn implementation assumes numerical or numpy array values and tests
them for equality.up to error tolerance specified by --tol option.

See _demo_client for an example of using the debug server.
To test this script run these processes in seperate terminals:
    
python debug_test.py 
python debug_test.py --client
python debug_test.py --client
'''

import numpy as np
import socket
import sys
import struct
import cPickle

from argparse import ArgumentParser

sock = None
local_data = {}
ACK = b'0'
ERR = b'1'

default_ip = '127.0.0.1'
default_port = 6666

def _send_msg(sock, msg):    
    pkl = cPickle.dumps(msg)
    sock.send(struct.pack("i", len(pkl)) + pkl)
    
def _recv_msg(sock):      
    int_size = struct.calcsize("i")
    
    if len(sock.recv(int_size, socket.MSG_PEEK)) < int_size:
        return None
    
    size = struct.unpack("i", sock.recv(int_size))[0] 
    
    raw = sock.recv(size, socket.MSG_WAITALL)        
    return cPickle.loads(raw)
    
def connect_to_server(server_ip=default_ip, server_port=default_port):
    global sock
    
    if sock is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print >> sys.stderr, '[DBG_CLIENT] Connecting to server %s:%d ... ' % (server_ip, server_port),
        sock.connect((server_ip, server_port))    
        print 'Connection Established'
    else:
        print >> sys.stderr, '[DBG_CLIENT] Already connected to server. Ignoring connection request.'
        
def send_data(label, data):
    local_data[label] = data

def check_point(label):    
    _send_msg(sock, (label, local_data))
    print >> sys.stderr, '[DBG_CLIENT] Reached checkpoint (%s) ... ' % label,
    reply = sock.recv(1, socket.MSG_WAITALL)
    
    if reply == ACK:
        print >> sys.stderr, 'pass'        
    else:
        remote_data = _recv_msg(sock)
        print >> sys.stderr, 'fail'
        import pdb; pdb.set_trace()

    local_data.clear()

def _demo_client(args):
    connect_to_server(args.server_ip, args.port)
    
    # This part produces idnetical output
    np.random.seed(0)
    a = np.random.rand(5)    
    b = np.random.rand(5)    
    send_data('a', a)
    send_data('b', b)
    check_point('step_1')
    
    # This part can produce different outputs
    np.random.seed(None)
    c = np.random.rand(5)    
    d = np.random.rand(5)    
    send_data('k', 0) # Match
    send_data(str(np.random.randint(100)), c) # Different keys
    send_data('d', d)                    # Same key, different values
    check_point('step_2')    
       
def _accept_connetions(args): 
    num_clients = 1 if args.single else 2       
    ssock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if args.reuse:
        ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    ssock.bind((args.server_ip, args.port))
    print >> sys.stderr, 'Starting debug server on port %d' % args.port    
    ssock.listen(1)
        
    connection = [None] * num_clients
    client_address = [None] * num_clients
    
    for i in xrange(num_clients):
        connection[i], client_address[i] = ssock.accept()
        print >> sys.stderr, 'Client %i connected' % i 
                        
    ssock.close()
    return connection, client_address

def _check_data(lbl0, lbl1, data0, data1, args):
    if lbl0 != lbl1:
        print >> sys.stderr, 'Checkpoint label mismatch'
        return False
    
    s0 = set(data0.keys())
    s1 = set(data1.keys())
    
    key_diff01 = set(s0)-set(s1)
    key_diff10 = set(s1)-set(s0)
    
    chk_pass = True
    
    if len(key_diff01) > 0:
        print >> sys.stderr, 'The following keys from client 0 has no match from client 1:'
        for x in key_diff01: print x
        chk_pass = False
        
    if len(key_diff10) > 0:
        print >> sys.stderr, 'The following keys from client 1 has no match from client 0:'
        for x in key_diff10: print x
        chk_pass = False
        
    for k in (s0&s1):
        v0 = np.array(data0[k])
        v1 = np.array(data1[k])
                            
        if v0.shape != v1.shape: 
            print >> sys.stderr, 'Shape mismatch for key %s' % k
            chk_pass = False     
        else:
            diff = np.max(np.abs(v0-v1))
            
            if diff > args.tol:
                print >> sys.stderr, 'ERROR: Value mismatch for key %s exceeds threshold (max error = %e)' % (k, diff)
                chk_pass = False    
            elif diff > 0.0:
                print >> sys.stderr, 'WARNING: Non-zero value difference for key %s (max error = %e)' % (k, diff)            
                                        
    return chk_pass

def _run_server(args):        
    num_clients = 1 if args.single else 2
    connection, client_address = _accept_connetions(args)    
    
    while True:
        rec_flag = [0] * num_clients
        data = [None] * num_clients
        lbl = [None] * num_clients
        
        while sum(rec_flag) < num_clients:
            for i in xrange(num_clients):        
                msg = _recv_msg(connection[i])
            
                if msg:
                    print >> sys.stderr, 'Recieved checkpoint (%s) from client %d' % (msg[0], i)
                    rec_flag[i] = 1
                    lbl[i] = msg[0]
                    data[i] = msg[1]
            
        chk = True if args.single else _check_data(lbl[0], lbl[1], data[0], data[1], args)    
        print >> sys.stderr, 'Checkpoint passed' if chk else 'Checkpoint failed'
        
        if chk:
            for c in connection: c.send(ACK)            
        else:
            connection[0].send(ERR)
            _send_msg(connection[0], data[1])
            connection[1].send(ERR)
            _send_msg(connection[1], data[0])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--server_ip', type=str, default=default_ip)
    parser.add_argument('--port', type=int, default=default_port)
    parser.add_argument('--client', action='store_true', help='Runs a demo client')
    parser.add_argument('--reuse', action='store_true', help='Sets SO_REUSEADDR socket option.')    
    parser.add_argument('--tol', type=float, default=0.0, help='Error tolerance')
    parser.add_argument('--single', action='store_true', help='Accepts a single client. No comparisons are made.')
              
    args = parser.parse_args()        
    
    if args.client:
        _demo_client(args)
    else:
        _run_server(args)
    
