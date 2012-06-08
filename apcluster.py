#!/usr/bin/env python

import sys
#from numpy import array, diag, zeros, ones_like, identity, amax, maximum, amin, minimum, loadtxt, kron, ones, reshape
import numpy as np
from numpy.random import rand

#from matplotlib.pyplot import *

from optparse import OptionParser

usage = '%prog [options] <similarity matrix file>'
desc = """\
Python implementation of the Affinity Propagation algorithm by Frey et al
(http://www.psi.toronto.edu/index.php?q=affinity%20propagation). Accepts a
file containing a similarity matrix of all pair-wise comparisons between
a set of objects.  File should be formatted so the numpy.loadtxt() function
can load it.  Self similarity scores can be set either explicitly or as a
fraction of the data range in the matrix itself.  Generally speaking, small
absolute self-similarity scores (or a ratio of 1) give the fewest clusters,
and large scores or a ratio of 0 gives many clusters. No self-similarity
adjustment is made by default (diagonal values of matrix are used as-is).
"""
parser = OptionParser(usage=usage,description=desc)
parser.add_option('-r','--ratio',dest='ratio',type='float',default=None,
                  help='self similarity values all set to (1-<r>)*<max value>+<r>*<min_value> in matrix')
parser.add_option('-a','--absolute',dest='abso',type='float',default=None,
                  help='absolute self similarity value to set for all, incompatible with --ratio')
parser.add_option('-i','--interactive',dest='inter',action='store_true',
                  help='run in interactive mode')

def ap_cluster(S) :

    n = S.shape[0]

    # this was in original matlab code, might not be necessary
    realmax = np.finfo(float).max
    realmin = np.finfo(float).tiny
    eps = np.finfo(float).eps
    S = S + (eps*S+realmin*100)*rand(n,n)
    A = np.zeros((n,n))
    R = np.zeros((n,n))
    lam = 0.5

    for itr in range(100) :

        # Compute responsibilities
        Rold = R
        AS = A + S
        Y, I = AS.max(1), AS.argmax(1)
        for i in range(n) :
            AS[i,I[i]] = -realmax
        Y2, I2 = AS.max(1), AS.argmax(1)
        R = S - np.kron(np.ones((n,1)),Y).T
        for i in range(n) :
            R[i,I[i]] = S[i,I[i]]-Y2[i]
        # Dampen responsibilities
        R = (1-lam)*R + lam*Rold

        # Compute availabilities
        Aold = A
        Rp = np.maximum(R,0)
        for i in range(n) :
            Rp[i,i] = R[i,i]
        A = np.kron(np.ones((n,1)),Rp.sum(0)) - Rp
        dA = np.diag(A)
        A = np.minimum(A,0)
        for i in range(n) :
            A[i,i] = dA[i]
        A = (1-lam)*A + lam*Aold

    E = R.T + A
    end_I = np.where(np.diag(E)>0)[0]
    K = end_I.size
    tmp, c = S[:,end_I].max(1), S[:,end_I].argmax(1)
    c[end_I] = np.arange(K)
    idx = end_I[c]

    return E, R, A, end_I, c, idx


def set_diagonal(S,val) :

    n = S.shape[0]

    # when ubuntu repo freaking updates to numpy 1.4.0 we can use this
    #fill_diagonal(S_orig,self_sim)
    # ensure diagonal is zero
    S *= np.ones_like(S)-np.identity(n)
    # make diag(S_orig) = self_sim
    S += np.identity(n)*val

    return S

if __name__ == '__main__' :

    opts, args = parser.parse_args(sys.argv[1:])

    if len(args) != 1 :
        parser.error('Exactly 1 non-option argument is required')

    if opts.ratio is not None and opts.abso is not None :
        parser.error('--ratio and --absolute options are incompatible, specify one')

    # the similarity matrix, now a numpy array
    S_orig = np.loadtxt(args[0])

    if opts.ratio is not None :
        S_min, S_max = np.amin(S_orig), np.amax(S_orig)
        self_sim_ratio = opts.ratio
        self_sim = self_sim_ratio*S_min + (1-self_sim_ratio)*S_max
    elif opts.abso is not None :
        self_sim = opts.abso
    else :
        self_sim = None

    if self_sim is not None :
        S_orig = set_diagonal(S_orig,self_sim)

    E, R, A, end_i, c, idx = ap_cluster(S_orig)

    sorted_clusters = [(list(idx).count(x),x) for x in end_i]
    sorted_clusters.sort(reverse=True)

    for i,(c_cnt, exem) in enumerate(sorted_clusters) :
        print 'cluster %d for exemplar %d'%(i,exem)
        idxs = filter(lambda x: x[1] == exem, zip(range(len(idx)),idx))
        print [x[0] for x in idxs]

    # print biggest clusters first

    #subplot(1,2,1)
    #pcolor(concatenate((c.reshape(n,1),c.reshape(n,1)),axis=1))
    #colorbar()

    #subplot(1,2,2)
    #pcolor(S_orig)
    #colorbar()

    #show()
