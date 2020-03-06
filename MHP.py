##########################

# Implementation of MAP EM algorithm for Hawkes process
#  described in:
#  https://stmorse.github.io/docs/orc-thesis.pdf
#  https://stmorse.github.io/docs/6-867-final-writeup.pdf
# For usage see README
# For license see LICENSE
# Author: Steven Morse
# Email: steventmorse@gmail.com
# License: MIT License (see LICENSE in top folder)

##########################


import numpy as np
import time as T

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import cartesian

import matplotlib.pyplot as plt
import math

class MHP:
    def __init__(self, alpha=[0.5], mu=[0.1], omega=1.0, I = 4, M = 6, N = 120 ):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''
        
        self.data = []
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0] # dimension
        # self.check_stability()
        self.actors = I
        self.actor_pairs = M
        self.nEvent = N


    def actor_p(self):
        '''
        I : the number of actors
        M : the number of actors-pairs
        '''
        self.actor_pair = []
        pair = []
        number = self.actors + 1
        for i in range(1,number):
            for j in range(1,number):
                if i < j:
                    pair = [i, j]
                    if pair not in self.actor_pair and len(self.actor_pair) < self.actor_pairs:
                        self.actor_pair.append(pair)


    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w,v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')
    
    def get_sum(self, i, j):
        if(self.actor_pair[i][0] != self.actor_pair[j][0] and self.actor_pair[i][0] != self.actor_pair[j][1] and self.actor_pair[i][1] != self.actor_pair[j][0] and self.actor_pair[i][1] != self.actor_pair[j][1]):
            self.result = 0
        else:
            self.result = 1
        return self.result


    def generate_seq(self):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''

        self.data = []  # clear history

        Istar = np.sum(self.mu)
        
        #s = np.random.exponential(scale=1./Istar)
        s = 1.0

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim), 
                              1, 
                              p=(self.mu / Istar))
        self.data.append([s, n0])

        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        lastrates = self.mu.copy()

        decIstar = False
        while True:
            tj, uj = self.data[-1][0], int(self.data[-1][1])
            

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
            else:
                # otherwise, we just had an event, so recalc Istar (inclusive of last event)
                sum_alpha = 0
                for i in range(self.dim):
                    sum_alpha += self.alpha[i] * self.get_sum(i, uj)
                Istar = np.sum(lastrates) + \
                        self.omega * sum_alpha
            # print(Istar)

            # generate new event
            s += np.random.exponential(scale=1./Istar)
            # print(s)
            rates = []

            # calc rates at time s (use trick to take advantage of rates at last event)
            for i in range(len(self.mu)):
                rate = self.mu[i] + np.exp(-self.omega * (s - tj)) * \
                    (self.alpha[i] * self.get_sum(i, uj) * self.omega + lastrates[i] - self.mu[i])
                rates.append(rate)
            # print(rates)

            # rates = self.mu + np.exp(-self.omega * (s - tj)) * \
            #         (self.alpha[:,uj].flatten() * self.omega + lastrates - self.mu)

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            # print(Istar)
            # print(np.sum(rates))
            # print(diff)
            try:
                n0 = np.random.choice(np.arange(self.dim+1), 1, 
                                        p=(np.append(rates, diff) / Istar))
            except ValueError:
                # by construction this should not happen
                print('Probabilities do not sum to one.')
                self.data = np.array(self.data)
                return self.data

            if n0 < self.dim:
                self.data.append([s, n0])
                # update lastrates
                lastrates = rates.copy()
                # print('1')
                
            else:
                decIstar = True
                # print('0')

            # if past nEvent, done
            if len(self.data) == self.nEvent:
                self.data = np.array(self.data)
                return self.data


    #-----------
    # EM LEARNING
    #-----------

    def EM(self, Ahat, mhat, omega, N, M, seq=[], smx=None, tmx=None, regularize=False, 
           Tm=-1, maxiter=100, epsilon=0.00001, verbose=True):
        '''implements MAP EM. Optional to regularize with `smx` and `tmx` matrix (shape=(dim,dim)).
        In general, the `tmx` matrix is a pseudocount of parent events from column j,
        and the `smx` matrix is a pseudocount of child events from column j -> i, 
        however, for more details/usage see https://stmorse.github.io/docs/orc-thesis.pdf'''
        
        # if no sequence passed, uses class instance data
        if len(seq) == 0:
            seq = self.data

        N = len(seq)
        dim = mhat.shape[0]
        Tm = float(seq[-1,0]) if Tm < 0 else float(Tm)
        sequ = seq[:,1].astype(int)

        eta_nn = np.random.uniform(0.01, 0.99, size=N)
        eta_ln = np.random.uniform(0.01, 0.99, size=(N, N))
        pi = np.random.randint(0, high=2, size=(N, M))

        # PRECOMPUTATIONS

        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        diffs = pairwise_distances(np.array([seq[:,0]]).T, metric = 'euclidean')
        diffs[np.triu_indices(N)] = 0

        # kern[i,j] = omega*np.exp(-omega*diffs[i,j])
        kern = omega*np.exp(-omega*diffs)


        k = 0
        old_LL = -10000
        START = T.time()
        while k < 21:

            # get Sm to compute pi
            # get_sm : return the list of event index that share actors with i
            # i -- dimension
            def get_sm(i):
                sm = []
                for k in range(len(self.actor_pair)):
                    if(self.actor_pair[i][0] == self.actor_pair[k][0] or self.actor_pair[i][0] == self.actor_pair[k][1] or self.actor_pair[i][1] == self.actor_pair[k][0] or self.actor_pair[i][1] == self.actor_pair[k][1]):
                        sm.append(k)
                return sm

            # get_sum_sm : return the sum of pi
            # i -- dimension, j -- the index of event
            def get_sum_sm(i, j):
                sm = get_sm(i)
                res_sum = 0
                for m in sm:
                    res_sum += pi[j, m]
                return res_sum

            # compute pi
            def get_pi(n,m):
                pi_result = 0
                
                # self-trigger
                trigger = 0
                try:
                    trigger = eta_nn[n] * np.log(mhat[m])
                except:
                    print('log error!')
                pi_result += trigger

                # time-decay effect
                sm = get_sm(m)
                time = 0
                for m in sm:
                    time += Ahat[m] 
                pi_result += time

                # from the past
                sum_past = 0
                try:
                    for l in range(n-1):
                        if seq[l,1] == m:
                            if eta_ln[n, l] == 0.0:
                                eta_ln[n, l] = 0.000000000000000001
                            sum_past += eta_ln[n,l] * get_sum_sm(m, l) * np.log(Ahat[m] * kern[n,l] / eta_ln[n,l])
                except:
                    print('log error!')
                pi_result += sum_past

                # to future
                sum_future = 0
                try:
                    for l in range(n+1,N):
                        if seq[l,1] == m:
                            sum_future += eta_ln[l,n] * get_sum_sm(m, l) * np.log(Ahat[m] * kern[l,n]) 
                except:
                    print('log error!')
                pi_result += sum_future
                
                return pi_result
                 

            # update pi
            for i in range(len(sequ)):
                if sequ[i] != -1:
                    pi[i,:] = 0
                    pi[i, sequ[i]] = 1
                else:
                    sum_pi = 0
                    for j in range(dim):  
                        pi[i, j] = get_pi(i,j)
                        sum_pi += pi[i,j]             
                    for j in range(dim):
                        pi[i,j] = pi[i,j] / sum_pi
                    pi = np.array(pi, dtype = 'float')   

            # Au
            Au = Ahat[sequ]
            ag = np.multiply(Au, kern)
            ag[np.triu_indices(N)] = 0
            ag_pi = ag.copy()

            # compute m_{u_i}
            mu = mhat[sequ]

            # compute total rates of u_i at time i
            rates = mu + np.sum(ag, axis=1)
           

            # compute ag with pi
            for i in range(0, ag.shape[0]):
                for j in range(0, ag.shape[1]):
                    dimension = sequ[j]
                    ag_pi[i][j] = get_sum_sm(dimension,j) * ag[i,j]
            
            # compute rate with pi
            rate_pi = mu + np.sum(ag_pi, axis=1)

            # compute matrix of eta_nn and eta_ln  (keep separate for later computations)
            eta_ln = np.divide(ag, np.tile(np.array([rate_pi]).T, (1,N)))
            eta_nn = np.divide(mu, rate_pi)
            #print('eta done!')

            # compute mhat:  mhat_u = (\sum_{u_i=u} eta_nn) / T
            
            mhat = []
            for i in range(dim):
                seq_idx = np.where(seq[:,1] == i)[0]
                m_sum = 0
                pi_beta = pi[:,i]
                for j in seq_idx:
                    m_sum += eta_nn[j] * pi_beta[j]
                    if m_sum == 0:
                        m_sum == 0.0000000000001
                mhat.append(m_sum)
            mhat = np.array(mhat)
            mhat /= Tm
            # print('mhat done!')

            # returns sum of all pmat vals where u_i=a
            # *IF* pmat upper tri set to zero, this is 
            # \sum_{u_i=u}\sum_{u_j=u', j<i} p_{ij}
            def sum_etaln(a):
                c = np.where(seq[:,1]==int(a))[0] # record the row index of event on every dimension
                sum_l = 0
                sum_n = 0
                for n in c:
                    for l in range(n-1):
                        if l in c:
                            sum_l += eta_ln[n, l] # ignore get_sum_sm(a,l)
                    sum_n += pi[n, a] * sum_l
                return sum_n
            # vp = np.vectorize(sum_etaln)

            # approx of Gt sum in a_{uu'} denom
            seqcnts = []
            seq_idx = []
            for i in range(dim):
                seq_idx = np.where(seq[:,1] == i)[0]
                pi_beta = pi[:,i]
                alpha_sum = 0
                for j in seq_idx:
                    alpha_sum += pi_beta[j]
                seqcnts.append(alpha_sum)
            seqcnts = np.array(seqcnts)
            # print('seqcnts done!')

            # approximate with G(T-T_j) = 1
            vp = []
            for i in range(dim):
                vp_res = sum_etaln(i)
                vp.append(vp_res)
            # print(vp)
            # print('vp done!')

            Ahat = np.divide(np.array(vp),seqcnts)
            # print(Ahat)
            # print('ahat done!')
            
            def get_term11(n,m):
                c = np.where(seq[:,1]==int(m))[0] # record the row index of event on every dimension
                sum_l = 0
                try:
                    for l in range(n-1):
                        if l in c:
                            # print(Ahat[m]*kern[n,l])
                            if eta_ln[n, l] == 0.0:
                                eta_ln[n,l] = 0.0000000000001
                            sum_l += get_sum_sm(m, l) * eta_ln[n, l] * math.log(Ahat[m]*kern[n,l]/eta_ln[n, l])
                except ValueError:
                    print('value error in term11!')
                return sum_l
            
            def get_term12(n,m):
                #sum_eta = 0
                try:
                    if eta_nn[n] == 0.0:
                        eta_nn[n] = 0.00000000001
                    sum_eta = eta_nn[n] * math.log(mhat[m]/eta_nn[n])
                except ValueError:
                    print('log error in term12!')
                return sum_eta

            if k % 10 == 0:
                #term1 = np.sum(np.log(rates))
                term1 = 0
                for i in range(N):
                    for j in range(dim):
                        term11 = get_term11(i,j)
                        term12 = get_term12(i,j)
                        term1 += pi[i][j]*(term11 + term12)
                term2 = Tm * np.sum(mhat)
                term3 = 0
                for i in range(dim):
                    for j in range(N):
                        term3 += get_sum_sm(i, j) * Ahat[i]
                new_LL = (1./N) * (term1 - term2 - term3)
                # new_LL = (1./N) * (term1 - term3)
                if abs(new_LL - old_LL) <= epsilon:
                    if verbose:
                        print('Reached stopping criterion. (Old: %1.3f New: %1.3f)' % (old_LL, new_LL))
                    return Ahat, mhat, pi
                if verbose:
                    print('After ITER %d (old: %1.3f new: %1.3f)' % (k, old_LL, new_LL))
                    print(' terms %1.4f, %1.4f, %1.4f' % (term1, term2, term3))

                old_LL = new_LL

            k += 1

        if verbose:
            print('Reached max iter (%d).' % maxiter)

        self.Ahat = Ahat
        self.mhat = mhat
        self.pi = pi
        return Ahat, mhat, pi

    #-----------
    # VISUALIZATION METHODS
    #-----------
    
    def get_rate(self, ct, d):
        # return rate at time ct in dimension d
        seq = np.array(self.data)
        if not np.all(ct > seq[:,0]): seq = seq[seq[:,0] < ct]
        return self.mu[d] + \
            np.sum([self.alpha[d,int(j)]*self.omega*np.exp(-self.omega*(ct-t)) for t,j in seq])

    def plot_rates(self, horizon=-1):

        if horizon < 0:
            horizon = np.amax(self.data[:,0])

        f, axarr = plt.subplots(self.dim*2,1, sharex='col', 
                                gridspec_kw = {'height_ratios':sum([[3,1] for i in range(self.dim)],[])}, 
                                figsize=(8,self.dim*2))
        xs = np.linspace(0, horizon, (horizon/100.)*1000)
        for i in range(self.dim):
            row = i * 2

            # plot rate
            r = [self.get_rate(ct, i) for ct in xs]
            axarr[row].plot(xs, r, 'k-')
            axarr[row].set_ylim([-0.01, np.amax(r)+(np.amax(r)/2.)])
            axarr[row].set_ylabel('$\lambda(t)_{%d}$' % i, fontsize=14)
            r = []

            # plot events
            subseq = self.data[self.data[:,1]==i][:,0]
            axarr[row+1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'bo', alpha=0.2)
            axarr[row+1].yaxis.set_visible(False)

            axarr[row+1].set_xlim([0, horizon])

        plt.show()


    def plot_events(self, horizon=-1, showDays=True, labeled=True):
        if horizon < 0:
            horizon = np.amax(self.data[:,0])

        fig = plt.figure(figsize=(10,2))
        ax = plt.gca()
        for i in range(self.dim):
            subseq = self.data[self.data[:,1]==i][:,0]
            plt.plot(subseq, np.zeros(len(subseq)) - i, 'bo', alpha=0.2)

        if showDays:
            for j in range(1,int(horizon)):
                plt.plot([j,j], [-self.dim, 1], 'k:', alpha=0.15)

        if labeled:
            ax.set_yticklabels('')
            ax.set_yticks(-np.arange(0, self.dim), minor=True)
            ax.set_yticklabels([r'$e_{%d}$' % i for i in range(self.dim)], minor=True)
        else:
            ax.yaxis.set_visible(False)

        ax.set_xlim([0,horizon])
        ax.set_ylim([-self.dim, 1])
        ax.set_xlabel('Days')
        plt.show()

