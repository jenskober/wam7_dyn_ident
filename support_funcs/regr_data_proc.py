import math
import numpy
import scipy.signal
import pickle


def butter_lfilter( N, Wn, signal):
    butter_b,butter_a = scipy.signal.butter( N, Wn )
    filtered_signal = scipy.signal.lfilter(butter_b,butter_a,signal)
    return filtered_signal


_inf = float('inf')

def butter_filtfilt( N, Wn, signal):
    if Wn == _inf: return signal[:]
    butter_b,butter_a = scipy.signal.butter( N, Wn )
    filtered_signal = scipy.signal.filtfilt(butter_b,butter_a,signal)
    return filtered_signal


def central_diff( array, div, n = 2 ):
    size = len( array )
    diff = numpy.zeros_like( array )
    if n == 1:
        diff[0] = ( array[1] - array[0]  ) / div
        for i in range(1,size-1):
            diff[i] = ( array[i+1] - array[i-1]  ) / (2*div)
        diff[size-1] = ( array[size-1] - array[size-2]  ) / div
    elif n == 2:
        diff[0] = ( array[1] - array[0]  ) / div
        diff[1] = ( array[2] - array[0]  ) / (2*div)
        for i in range(2,size-2):
            diff[i] = ( - array[i+2] + 8*array[i+1] - 8*array[i-1] + array[i-2] ) / (12*div)
        diff[size-2] = ( array[size-1] - array[size-3]  ) / (2*div)
        diff[size-1] = ( array[size-1] - array[size-2]  ) / div
    else:
        raise Exception('use n = 1 or 2')
    return diff


def central_2nd_diff( array, div, n = 2 ):
    size = len( array )
    diff = numpy.zeros_like( array )
    if n == 1:
        diff[0] = ( array[1+1] - 2*array[1] + array[1-1]  ) / (div*div)
        for i in range(1,size-1):
            diff[i] = ( array[i+1] - 2*array[i] + array[i-1]  ) / (div*div)
        diff[size-1] = ( array[(size-2)+1] - 2*array[(size-2)] + array[(size-2)-1]  ) / (div*div)
    elif n == 2:
        diff[0] =  diff[1] = ( array[1+1] - 2*array[1] + array[1-1]  ) / (div*div)
        for i in range(2,size-2):
            diff[i] = ( - array[i+2] + 16*array[i+1] - 30*array[i] + 16*array[i-1] - array[i-2] ) / (12*div*div)
        diff[size-1] = diff[size-2] = ( array[(size-2)+1] - 2*array[(size-2)] + array[(size-2)-1]  ) / (div*div)
    else:
        raise Exception('use n = 1 or 2')
    return diff


def read_data( dof, h, rbtlogfile, trajreffile ):
    
    rbtlog = numpy.loadtxt(rbtlogfile)
    s = rbtlog.shape[0]
    t = numpy.array(rbtlog[:,0])
    q = numpy.array(rbtlog[:,1:dof+1])
    tau = numpy.array(rbtlog[:,dof+1:dof*2+1])
    
    h_avg = (t[-1]-t[0])/len(t)
    #print ('h avg',h_avg,'h nom',h)
    if abs( ( h_avg / h ) - 1 ) > 10e-5 :
        print('h nom != h avg')
        print ('h avg',h_avg,'h nom',h)
      
    trajref = numpy.loadtxt(trajreffile)
    reft = numpy.array([ h*i for i in range(trajref.shape[0]) ])
    
    return t, q, tau, reft, trajref


def diff_and_filt_data( dof, h, q_raw, tau_raw, fc_q, fc_dq, fc_ddq, fc_tau):

    s = q_raw.shape[0]
    
    q = numpy.zeros_like(q_raw)
    dq = numpy.zeros_like(q_raw)
    ddq = numpy.zeros_like(q_raw)
    tau = numpy.zeros_like(tau_raw)
    wc_q = fc_q * 2 * math.pi * h
    wc_dq = fc_dq * 2 * math.pi * h
    wc_ddq = fc_ddq * 2 * math.pi * h
    wc_tau = fc_tau * 2 * math.pi * h

    for i in range(dof):
        q[:,i] = butter_filtfilt( 3, wc_q, q_raw[:,i] )
        
        joint_i_dq_raw = central_diff(q_raw[:,i],h,2)
        dq[:,i] = butter_filtfilt( 3, wc_dq, joint_i_dq_raw )
        #dq[:,i] = central_diff(q[:,i],h,2)
        
        joint_i_ddq_raw = central_diff( joint_i_dq_raw ,h,2)
        ddq[:,i] = butter_filtfilt( 3, wc_ddq, joint_i_ddq_raw )
        #ddq[:,i] = central_diff( central_diff(q[:,i],h,2) ,h,2)
        
        tau[:,i] = butter_filtfilt( 3, wc_tau, tau_raw[:,i] )
        # tau[:,i] = butter_lfilter( 3, wc_tau, tau_raw[:,i] )
    
    return q, dq, ddq, tau





def range_taus( range_link ):
    ti = range_link[0]-1
    tf = range_link[-1]
    return range(ti,tf)


def range_parms( range_link ):
    pi = rbt.rB[range_link[0]-1]
    pf = rbt.rB[range_link[-1]]
    return range(pi,pf)


def ident_matrices( parms, rbt, q, dq, ddq, tau, range_linktaus=None, range_linkparms=None ):
        
        parms = parms.lower()
        
        if range_linktaus == None:
            range_linktaus= (1,rbt.dof)
        if range_linkparms == None:
            range_linkparms= (1,rbt.dof)
        
        ti = range_linktaus[0]-1
        tf = range_linktaus[-1]
        
        tn = tf-ti
        
        if parms == 'all':
            pi = rbt.r[range_linkparms[0]-1]
            pf = rbt.r[range_linkparms[-1]]
        elif parms == 'effective':
            pi = rbt.rE[range_linkparms[0]-1]
            pf = rbt.rE[range_linkparms[-1]]
        elif parms == 'base':
            pi = rbt.rB[range_linkparms[0]-1]
            pf = rbt.rB[range_linkparms[-1]]
            
        pn = pf-pi
        
        s = q.shape[0]
        
        W = numpy.zeros( ( tn*s, pn ) )
        T = numpy.zeros( tn*s )
            
        if parms == 'all':
            for i in range(s):
                W[ i*tn : i*tn+tn , : ] = rbt_Y( q[i,:], dq[i,:], ddq[i,:] )[ ti:tf , pi:pf ]
        elif parms == 'effective':
            for i in range(s):
                W[ i*tn : i*tn+tn , : ] = ( matrix(rbt_Y( q[i,:], dq[i,:], ddq[i,:] )) * rbt.D_EY ).numpy()[ ti:tf , pi:pf ]
        elif parms == 'base':  
            for i in range(s):
                W[ i*tn : i*tn+tn , : ] = rbt_YB( q[i,:], dq[i,:], ddq[i,:] )[ ti:tf , pi:pf ] 
        
        for i in range(s):
            T[ i*tn : i*tn+tn ] = tau[i][ti:tf]    
        
        return W,T





def regr_matrices( dof, parm_num, q, dq, ddq, tau, regr_func):

  sn = q.shape[0]

  H_S = numpy.matrix( numpy.zeros( ( dof*sn, parm_num ) ) )
  tau_S = numpy.matrix( numpy.zeros( dof*sn ) ).T

  for i in range(sn):
    H_S[ i*dof : i*dof+dof , : ] = numpy.array( regr_func( q[i], dq[i], ddq[i] ) ).reshape(dof,parm_num)

  for i in range(sn):
      tau_S[ i*dof : i*dof+dof ] = numpy.asmatrix( tau[i] ).T

  return H_S,tau_S


import sympybotics as spb
from sympybotics._compatibility_ import exec_

def gen_regr_matrices(rbt, q, dq, ddq, tau):
  exec_(spb.robot_code_to_func('python', rbt.H_code, 'H', 'regressor_func', rbt.rbtdef), globals())
  global sin, cos, sign
  sin = numpy.sin
  cos = numpy.cos
  sign = numpy.sign
  
  H_S, omega = regr_matrices( rbt.dof, rbt.dyn.n_dynparms, q, dq, ddq, tau, regressor_func )
  W = numpy.matrix(H_S[:,rbt.dyn.base_idxs])
  
  Q1,R1 = numpy.linalg.qr(W)
  rho1 = Q1.T*omega
  del H_S

  return W, omega, Q1, R1, rho1