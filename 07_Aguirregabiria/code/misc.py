import numpy as np
import pandas as pd
import os
from linearmodels import PanelOLS
from linearmodels import PooledOLS
from linearmodels import RandomEffects
from linearmodels import FirstDifferenceOLS
from linearmodels.datasets import jobtraining
from statsmodels.datasets import grunfeld
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

def indfind(state_m1,states):
    for j in range(len(states)):
        identifier= states[j] == state_m1
        try:
            identifier = all(identifier)
        except:
            pass
        if identifier:
            return(j)

list_var = [[0,1],[0,2]]
def my_meshgrid(list_var):
    str_lhs = ','.join('x_'+str(i) for i in range(len(list_var)))
    str_rhs = ','.join('list_var['+str(i)+']' for i in range(len(list_var)))
    exec(str_lhs+ '= np.meshgrid(' + str_rhs + ')')
    # xx,yy,zz = np.meshgrid(y1,y2,z)
    str_rhs_2 = ','.join('x_'+str(i) +'.flatten()' for i in range(len(list_var)))
    # meshed = np.vstack((xx.flatten(),yy.flatten(),zz.flatten()))
    meshed = eval('np.vstack((' + str_rhs_2 + '))')
    return(meshed.T)

def gen_na_i(i,nplayer):
    na_i = list(range(nplayer))
    na_i.pop(i)
    return(na_i)

def find_a_ind(a,a_space):
    for j in range(len(a_space)):
        if all(a_space[j,:]== a):
            return(j)
#-------------------------------

def clogit(ydum,x,restx,namesb,print_output):
    cconvb = 1e-6
    myzero = 1e-16
    nobs = int(ydum.shape[0])
    nalt = int(max(ydum) + 1)
    npar = int(x.shape[1]/nalt)
    max_iter = 100
    if npar != len(namesb):
        print("ERROR: Dimensions of x and of names(b0) \nrows(namesb)  do not match ")
        return

    xysum = 0
    for j in range(nalt):
        xbuff = x[:,int(npar*j):int(npar*(j+1))]
        xysum = xysum +  np.dot(np.diag(ydum == j),xbuff)

    iter = 1
    criter = 10
    llike = -nobs
    b0 = np.zeros(int(npar))

    while (criter > cconvb) & (iter < max_iter):
        if (print_output==1):
            print(" \n")
            print("Iteration                = ", iter ,"\n")
            print("Log-Likelihood function  = ", llike ,"\n")
            print("Norm of b(k)-b(k-1)      = ", criter,"\n")
            print(" \n")

        phat = np.zeros((nobs,nalt))
        for j in range(nalt):
            phat[:,j] = np.dot(x[:,int(npar*j):int(npar*(j+1))],b0).flatten() + restx[:,j]

        phat = (phat.T - phat.max(1)).T
        phat = np.exp(phat)
        phat = (phat.T / phat.sum(1)).T

        # Computing xmean
        sumpx = np.zeros((nobs,1))
        xxm = 0
        llike = 0
        for j in range(nalt):
             xbuff = x[:,int(npar*j):int(npar*(j+1))]
             sumpx = sumpx +  np.dot(np.diag(phat[:,j]), xbuff)
             xxm   = xxm + np.dot(xbuff.T, np.dot(np.diag(phat[:,j]), xbuff))
             llike = llike + ( (ydum == j) * np.log( (phat[:,j] > myzero) * phat[:,j]  + (phat[:,j] <= myzero) * myzero  )).sum()

        d1llike = xysum.sum(0) - sumpx.sum(0)
        # d2llike = np.dot(sumpx.T,sumpx)
        # Computing gradient

        # d1llike = xysum - sumpx.sum(0)
        # @ Computing hessian @
        d2llike = - (xxm - np.dot(sumpx.T,sumpx) )

        # @ Gauss iteration @
        invertible = abs(np.linalg.det(d2llike))>0;
        if invertible==1:
            b1 = b0 - np.dot(np.linalg.inv(d2llike),d1llike)
            criter = np.sqrt(np.dot((b1-b0).T,(b1-b0)))
            b0 = b1
            iter = iter + 1
        else:
            b0 = np.zeros((npar,1))
            criter = 0
        if(invertible==1):
            Avarb  = np.linalg.inv(-d2llike)
            sdb    = np.sqrt(np.diag(Avarb))
            tstat  = b0/sdb

            numyj  = np.array( [ 1 * (ydum==j) for j in range(3)])
            logL0  = numyj*np.log(numyj/nobs)
            logL0[logL0==np.nan] = 0
            logL0 = logL0.sum()
            lrindex = 1 - llike/logL0
            if (print_output==1):
                print("---------------------------------------------------------------------")
                print("Number of Iterations     = ", iter)
                print("Number of observations   = ", nobs)
                print("Log-Likelihood function  = ", llike)
                print("Likelihood Ratio Index   = ", lrindex)
                print("---------------------------------------------------------------------")
                print("       Parameter         Estimate        Standard        t-ratios")
                print("                                         Errors" )
                print("---------------------------------------------------------------------")
                for j in range(npar):
                    print(namesb[j],b0[j],sdb[j],tstat[j])
                    print("---------------------------------------------------------------------")
    return([b0,Avarb, invertible])

def MPE_br(pin, eqm,nplayer,nalt,numstate,states,actions,betad,ftran,zfun,theta_0,lambda_i):
    # and returns the best response choice probabilities.
    B_na = np.zeros([nplayer,numstate,nplayer-1])
    for i in range(nplayer):
        na_i = gen_na_i(i,nplayer)
        # B_1 = pin[:,[1,2]]
        B_na[i,:,:] = pin[:,na_i]
        # exec('B_na_'+ str(i+1) + '= pin[:,[' + ','.join([str(ii) for ii in na_i])  +'] ]' )
    #Create choice specific values for each player given flow profit and guess of choice probabilities
    if eqm ==1:
        pass #Player 2's Belief  = Player 1's CCP if eqm==1
         #Player 1's Belief= Player 2's CCP  if eqm==1
    else:
        for i in range(nplayer):
            B_na[i,:,:] = np.dot(np.diag(lambda_i[i,:]),B_na[i,:,:])


    ZP_fun = np.empty([nplayer,nalt,numstate,2+nplayer])
    FP_fun = np.zeros([nplayer,nalt,numstate,numstate])
    e_P_fun = np.empty([nplayer,nalt,numstate])

    # B_ii_tmp = np.zeros([numstate,4])
    for i in range(nplayer):
        na_i = gen_na_i(i,nplayer)
        for a_ii in range(nalt):
            a_na_i_space = actions[:,na_i][actions[:,i] == a_ii,:]
            FP_tmp = np.zeros(ftran[0,:,:].shape)
            ZP_tmp = np.zeros(zfun[0,0,:,:].shape)
            for a_na_i in range(len(a_na_i_space)):
                action_each = np.zeros(nplayer)
                action_each[i] = a_ii
                a = a_na_i_space[a_na_i]
                tmp = np.ones(numstate)
                for j_i in range(len(na_i)):
                    tmp = tmp * (a[j_i] * B_na[i,:,j_i] + (1-a[j_i])* (1 - B_na[i,:,j_i]) )
                    action_each[na_i[j_i]] = a[j_i]

                a_ind = find_a_ind(action_each,actions)
                FP_tmp = FP_tmp + np.dot(np.diag(tmp),ftran[a_ind,:,:])
                ZP_tmp = ZP_tmp + np.dot(np.diag(tmp),zfun[i,a_ind,:,:])
                # B_ii_tmp[i,a_ii,:,a_na_i] = tmp
            FP_fun[i,a_ii,:] = FP_tmp
            ZP_fun[i,a_ii,:,:] = a_ii * ZP_tmp
            e_P_fun[i,a_ii,:] = 0.5772156649 - a_ii * np.log(pin[:,i]) - (1 -a_ii) * np.log(1 - pin[:,i])

    # e0_P1 = 0.5772156649 - np.log(1-P1)

    ZP_bar = np.zeros([nplayer,numstate,nplayer+2])
    e_bar = np.zeros([nplayer,numstate])
    FP_bar = np.zeros([nplayer,numstate,numstate])
    W_z_bar = np.zeros([nplayer,numstate,nplayer+2])
    W_e_bar = np.zeros([nplayer,numstate])

    v_fun = np.zeros([nplayer,nalt,numstate])
    for i in range(nplayer):
        # for a_ii in range(nalt):
        for a_ii in range(nalt):
            ZP_bar[i,:,:] = ZP_bar[i,:,:] + np.dot(np.diag(a_ii * pin[:,i] + (1 - a_ii) * (1 - pin[:,i]) ) , ZP_fun[i,a_ii,:,:] )
            e_bar[i,:] = e_bar[i,:] + np.dot(np.diag(a_ii * pin[:,i] + (1 - a_ii) * (1 - pin[:,i]) ) , e_P_fun[i,a_ii,:] )
            FP_bar[i,:,:] = FP_bar[i,:,:] +  np.dot(np.diag(a_ii * pin[:,i] + (1 - a_ii) * (1 - pin[:,i]) ) , FP_fun[i,a_ii,:,:] )

    for i in range(nplayer):
        W_z_bar[i,:,:] = np.dot(np.linalg.inv(np.eye(numstate) - betad * FP_bar[i,:,:]), ZP_bar[i,:,:])
        W_e_bar[i,:] = np.dot(np.linalg.inv(np.eye(numstate) - betad * FP_bar[i,:,:]), e_bar[i,:])
    for i in range(nplayer):
        for a_ii in range(nalt):
            v_fun[i,a_ii,:] = np.dot(ZP_fun[i,a_ii,:,:] + betad*np.dot(FP_fun[i,a_ii,:,:],W_z_bar[i,:,:]),theta_0[i,:]) + betad * np.dot(FP_fun[i,a_ii,:,:],W_e_bar[i,:])
    pout = np.zeros([numstate,nplayer])

    for i in range(nplayer):
        v_i = v_fun[i,:,:]
        v_i_min = v_i.min(0)
        p_i = v_i - v_i_min
        p_i = np.exp(p_i)
        p_i = p_i / p_i.sum(0)
        pout[:,i] = p_i[1,:]

    return(pout)

def sim_data(M,Tdata,nplayer,nalt,numstate,states,actions,CP_true,ftran):
    #Set initial conditions for each market:
    # ✅ Checked data simulation
    #We will assume that for t=1 (initial states):
    # states with (y_1t-1, y_2t-1) = (0,0) occur 25% of the time
    # states with (y_1t-1, y_2t-1) = (1,0) occur 25% of the time
    # states with (y_1t-1, y_2t-1) = (0,1) occur 25% of the time
    # states with (y_1t-1, y_2t-1) = (1,1) occur 25% of the time

    Is_grid = np.ones(len(states))/len(states)
    index_m1 = np.random.choice(len(states),p=Is_grid,size=(M))
    Initial_states = states[index_m1,:]

    State_Data_full = np.zeros([M,Tdata+1])
    Action_Data_full = np.zeros([M,Tdata])
    State_Data_full[:,0] = index_m1
    ftran_p = np.zeros([numstate,numstate])
    for a_i in range(len(actions)):
        a = actions[a_i]
        tmp = np.ones(numstate)
        for i in range(nplayer):
            tmp = tmp * (a[i] * CP_true[:,i] + (1-a[i])* (1 - CP_true[:,i]) )
        ftran_p = ftran_p + np.dot(np.diag(tmp),ftran[a_i,:,:])


    for m in range(M):
        for t in range(Tdata):
            state_m1 = int(State_Data_full[m,t])
            # CCP_state_m1 = CP_true[index_m1[m0,]]
            # State_Data_full[m0,T] = state_m1
            index_tmr = np.random.choice(numstate,p=ftran_p[state_m1,:])
            state_tmr = states[index_tmr,:]
            action_tdy = state_tmr[0:nplayer]
            a_ind = find_a_ind(action_tdy,actions)
            Action_Data_full[m,t] = a_ind
            State_Data_full[m,t+1] = index_tmr
    State_Data_full = State_Data_full[:,0:Tdata]
    return([State_Data_full,Action_Data_full])

def estimates_ccp(states,actions,nplayer,nalt,numstate,State_Data_full,Action_Data_full):
    # ✅ Checked
    # Estimating CCPs - unrestricted - imposing the stationarity in the dgp
    #      Also collect summary data for each time period
    # Estimate CCPs using Raw frequency
    # Place holder
    est_states = np.zeros(len(states))
    est_prob = np.zeros([len(states),len(actions)])
    for s_i in range(len(states)):
        est_states[s_i] = (State_Data_full==s_i).sum()
        for a_i in range(len(actions)):
            est_prob[s_i,a_i] = ((State_Data_full==s_i) * (Action_Data_full==a_i)).sum()
    p_u_joint = (est_prob.T / est_states.T).T
    #Bound pu in (0,1)
    p_u_joint[p_u_joint>0.999] = 0.999
    p_u_joint[p_u_joint<0.001] = 0.001
    p_u_joint[p_u_joint==np.inf] = 0.5
    #CCPs to be used in the test
    p_u = np.zeros([nplayer,nalt,numstate])

    for i in range(nplayer):
        for a_ii in range(nalt):
            p_u[i,a_ii,:] = p_u_joint[:,actions[:,i]==a_ii].sum(1)
    p_u[p_u>0.999] = 0.999
    p_u[p_u<0.001] = 0.001
    p_u[p_u==np.inf] = 0.5

    return([p_u_joint,p_u])


def estimates_ccp_restricted(states,actions,nplayer,nalt,numstate,State_Data_full,Action_Data_full,p_u):
    #we estimate a vector \theta for each value of Y_1
    # unique values of the variable Y_1
    uniques = [0,1]
    p1_r = np.zeros((0,int(numstate/2)))#we collect the restricted p estimates here
    p2_r = np.zeros((0,int(numstate/2)))
    keep = []
    State_Data = State_Data_full.flatten()
    State_Data = states[State_Data.astype(int)]
    Action_Data = Action_Data_full.flatten()
    Action_Data = actions[Action_Data.astype(int)]

    est_states = np.zeros(numstate)
    est_freq = np.zeros([nplayer,nalt,numstate])
    for s_i in range(len(states)):
        est_states[s_i] = (State_Data_full==s_i).sum()
        for i in range(nplayer):
            for a_ii in range(nalt):
                est_freq[i,a_ii,s_i] = ((State_Data_full.flatten()==s_i) * (Action_Data[:,i]==a_ii)).sum()

    Mdotk = est_states

    for y0 in range(nalt):
        #Find the points in the state space with (Y_1 = y0)
        indy0 = np.arange(numstate)[states[:,0]==y0]
        numk = indy0.sum()
        # state_data_y0 = State_Data[:,0]==0
        # ydata_y0 = Action_Data
        # zdata_y0 = state_ids
        # zdata_y0 = zdata_y0[state_data_y0]

        p_uhat = p_u[:,1,indy0] #the probability of enter

        M_k = Mdotk[indy0]
        M = len(State_Data_full)
        #This is a procedure to find the (\theta, \lambda) that  maximizes the constrained likelihood.
        # It takes as an input an initial guess of \lambda, say \lambda^0 and two points in space of excluded variable zstar and finds the MLE \theta^1.
        #It then takes \theta^1 as given and finds the associated maximizer \lambda^1. It continues in this fashion
        #to convergence.

        # Test:
        lam0=np.zeros(8)
        tolp = 1e-8
        it=0
        thetastar_in = np.array(range(numstate))/50
        thetastar_out = np.ones(numstate)
        lamstar_in = lam0
        numrest = lam0.shape[0]
        lamstar_out = np.ones(lam0.shape)
        crit = max(np.abs(np.hstack([thetastar_in,lamstar_in]) - np.hstack([thetastar_out,lamstar_out])))
        while(crit>=tolp):
        #First step: given a guess of \lambda, find the MLE of \theta
        #Guess of initial theta
            theta_in = thetastar_in
            theta_out = np.ones(theta_in.shape)
            theta_crit = np.max(np.abs(theta_in-theta_out))
            itt = 1
            while (theta_crit>tolp) & (itt<=5000):
                ptheta = np.exp(theta_in)/(1+np.exp(theta_in))
                pthetad1 = ptheta *(1-ptheta)
                llik2 = np.hstack([M_k/M,M_k/M])*ptheta*(1-ptheta)
                llik2 = -np.diag(llik2)

                gradltheta = np.hstack([M_k/M,M_k/M])*( np.hstack([p_uhat[0,:],p_uhat[1,:]])-ptheta)

                #create 8x20 matrix of derivatives of 8 constraints w.r.t. 20 parameters column by column
                gradctheta11 = ( -( ptheta[int(numstate/2+1)] - ptheta[int(numstate/2)] )*( (theta_in[2:int(numstate/2)]-theta_in[1] )/(theta_in[1]-theta_in[0])**2))
                gradctheta12 = (( ptheta[int(numstate/2+1)] - ptheta[int(numstate/2)] )*(  (theta_in[2:int(numstate/2)]-theta_in[0] )/(theta_in[1]-theta_in[0])**2))
                #[TODO] What is this?
                gradctheta1k = np.eye(numrest)*(-( ptheta[int(numstate/2+1)] - ptheta[int(numstate/2)] )*(  (1)/(theta_in[1]-theta_in[0])))
                gradctheta21 = (( pthetad1[int(numstate/2)] )*(  (theta_in[2:int(numstate/2)]-theta_in[2] )/(theta_in[1]-theta_in[0])))
                gradctheta22 = -(( pthetad1[int(numstate/2+1)] )*(  (theta_in[2:int(numstate/2)]-theta_in[0] )/(theta_in[1]-theta_in[0])))
                gradctheta2k = np.eye(numrest) * pthetad1[int(numstate/2+2):numstate]

                gradctheta = np.concatenate([np.matrix(gradctheta11).T,np.matrix(gradctheta12).T,gradctheta1k,np.matrix(gradctheta21).T,np.matrix(gradctheta22).T,gradctheta2k],axis=1)
                ctheta = ptheta[int(numstate/2+2):numstate] - ptheta[int(numstate/2)] - (ptheta[int(numstate/2+1)] - ptheta[int(numstate/2)] )*((theta_in[2:int(numstate/2)]-theta_in[0])/(theta_in[1]-theta_in[0]) )
                #DMatrix
                dmatrix = np.concatenate([np.concatenate([llik2,gradctheta.T],axis=1),
                np.hstack([gradctheta,np.zeros((numrest,numrest))])])

                #Check if dmatrix is invertible
                invertible = np.abs(np.linalg.det(dmatrix))>0
                if invertible==1:
                    gradtheta = np.concatenate([ np.array(gradltheta+np.dot(gradctheta.T,lamstar_in)).flatten(),ctheta],axis=0)
                    invd_grad  = np.asarray(np.dot(np.linalg.inv(dmatrix),gradtheta)).flatten()
                    theta_out = theta_in - invd_grad[0:numstate]
                    theta_out = theta_in - invd_grad[0:numstate]
                    theta_crit = np.max(np.abs(theta_in-theta_out))
                    theta_in = theta_out
                    itt = itt+1
                else:
                    theta_crit = 0
            thetastar_out = theta_out
            if (invertible==1)& (itt<=5000):
                #Second step: given last estimate of \theta find the MLE of \lambda
                lamstar_out = lamstar_in - invd_grad[numstate:numstate+numrest]
                #Check for convergence and update estimates
                crit = max(np.abs(np.hstack([thetastar_in,lamstar_in]) - np.hstack([thetastar_out,lamstar_out])))
                lamstar_in  = lamstar_out
                thetastar_in = thetastar_out
                it = it+1
            else:
                crit=0
                thetastarlamstar = np.zeros(numstate+numrest)
        thetastarlamstar = np.hstack([thetastar_out,lamstar_out])
        converged = itt<=5000
        estimates = thetastarlamstar
        keep_y0=invertible
        converge = converged
            # return([thetastarlamstar, invertible, converged])

        # [estimates, keep_y0, converge] = cloglike(zeros(8,1))

        theta_out = estimates[0:numstate]
        ptheta = np.exp(theta_out)/(1+np.exp(theta_out))
        p1_r = np.vstack([p1_r,ptheta[0:int(numstate/2)]])
        p2_r = np.vstack([p2_r,ptheta[int(numstate/2):numstate]])

        keep.append(keep_y0)
    # End of for y0 in range(2)
    p1_r = p1_r.flatten()
    p2_r = p2_r.flatten()
    p1_r[np.isnan(p1_r)] = 0.5
    p2_r[np.isnan(p2_r)] = 0.5
    #Bound p_r in (0,1)
    p1_r[p1_r > 0.999] = 0.999
    p1_r[p1_r < 0.001] = 0.001
    p2_r[p2_r > 0.999] = 0.999
    p2_r[p2_r < 0.001] = 0.001

    p_r = np.zeros([nplayer,nalt,numstate])
    p_r[0,0,:] = p1_r
    p_r[0,1,:] = 1- p1_r
    p_r[1,0,:] = p2_r
    p_r[1,1,:] = 1- p2_r


    #Calculate LRT
    lrt_sim0 = 0
    for i in range(nplayer):
        for a_ii in range(nalt):
            lrt_sim0 = lrt_sim0 + 2 * est_freq[i,a_ii,:] * np.log(p_u[i,a_ii,:]) - np.log(p_r[i,a_ii,:])
    # if min(keep)==1:
        #mdotks = mdotks~mdotk;
    return([p_r,lrt_sim0,keep,converge])


def payoffs(B2_est, CP1_est,nplayer,nalt,numstate,states,actions,betad,ftran,zfun,theta_0,state_ids,actions1,Z1_1e,Z1_0e):
    #NPL Loop begins here
    # ✅ checked estimation
    # 🍋 Need to add three players
    k0 = 1
    Bin = B2_est
    Bout = np.ones(numstate)
    CP1 = CP1_est
    CP1_out = Bout
    crit = max(np.abs(CP1-CP1_out))
    tol = 1e-12

    while (crit>tol)& (k0<=1):
        ZP1 = np.dot(np.diag(Bin),Z1_1e) + np.dot(np.diag(1-Bin),Z1_0e)
        e0_P1 = 0.5772156649 - np.log(1-CP1)
        e1_P1 = 0.5772156649 - np.log(CP1)
        zbarP1 = np.dot(np.diag(CP1),ZP1)
        ebarP1 = CP1*e1_P1 + (1-CP1)*e0_P1

        ZP_1 = np.zeros([nalt,numstate,Z1_1e.shape[1]])
        ZP_1[0] = np.zeros(ZP1.shape)
        ZP_1[1] = ZP1
        FP_1 = np.zeros([nalt,numstate,numstate])
        FBar = np.zeros([numstate,numstate])
        for a_iii in range(nalt):
            FP_tmp = np.zeros(ftran[0,:,:].shape)
            ZP_tmp = np.zeros(zfun[0,0,:,:].shape)
            for a_ii in range(nalt):
                a_i = find_a_ind([1,a_ii], actions)
                FP_tmp = FP_tmp + np.dot(a_ii * Bin + (1-a_ii)*(1-Bin),ftran[a_i,:,:])
            FP_1[a_iii,:,:] = FP_tmp
            FBar = FBar + np.dot(a_iii * CP1 + (1-a_iii)*(1-CP1),FP_tmp)

        # FP_1_1 = np.dot(np.diag(Bin),ftran_11) + np.dot(np.diag(1-Bin),ftran_10)
        # FP_1_0 = np.dot(np.diag(Bin),ftran_01) + np.dot(np.diag(1-Bin),ftran_00)
        # FbarP_1 = np.dot(np.diag(CP1),FP_1_1) + np.dot(np.diag(1-CP1),FP_1_0)
        W1Pz = np.dot(np.linalg.inv(np.eye(FBar.shape[0])-betad*FBar),zbarP1)
        W1Pe =  np.dot(np.linalg.inv(np.eye(FBar.shape[0])-betad*FBar ),ebarP1)


        ztilde = np.zeros([nalt,numstate,zbarP1.shape[1]])
        etilde = np.zeros([nalt,numstate])

        for a_iii in range(nalt):
            ztilde[a_iii,:,:] = ZP_1[a_iii,:,:] + betad * np.dot(FP_1[a_iii,:,:],W1Pz)
            etilde[a_iii,:] = betad * np.dot(FP_1[a_iii,:,:],W1Pe)
        # ztilde1 = ZP1 + betad*np.dot(FP_1_1,W1Pz)
        # ztilde0 = np.zeros(zbarP1.shape) + betad*np.dot(FP_1_0,W1Pz)
        # etilde1 = betad*np.dot(FP_1_1,W1Pe)
        # etilde0 = betad*np.dot(FP_1_0,W1Pe)

        #Assign ztildes and etildes to data
        x = np.zeros([0,0])
        restx = np.zeros([0,0])
        for a_iii in range(nalt):
            if len(x) ==0:
                x = ztilde[a_iii,state_ids.astype(int),:]
                restx = etilde[a_iii,state_ids.astype(int)]

            else:
                x = np.hstack([x,ztilde[a_iii,state_ids.astype(int),:]])
                restx = np.vstack([restx.T,etilde[a_iii,state_ids.astype(int)]]).T
        # x = np.hstack([ztilde[0,state_ids.astype(int),:],ztilde[1,state_ids.astype(int),:]])
        # restx = np.vstack([etilde0[state_ids.astype(int)],etilde1[state_ids.astype(int)]]).T
        ydum = actions1
        # ydum = abs(ydum-2)
        namesb = ["alp1","del1","theta1"]
        [best, varest, convergence] = clogit(ydum,x,restx,namesb,0)
        if (convergence):
            #Update Belief Estimates
            kern = np.dot((ztilde[1,:,:]-ztilde[0,:,:]),best) + etilde[1,:]-etilde[0,:,]
            CP1_out = 1/(1+np.exp(-kern))
            ctilde = betad*np.dot(FP_1[1,:,:],np.dot(W1Pz,best))-betad*np.dot(FP_1[0,:,:],np.dot(W1Pz,best)) +  etilde[1,:]-etilde[0,:,]
            LHS = np.log(CP1_out/(1-CP1_out)) - ctilde
            pay1est = np.dot(Z1_1e,best)
            pay0est = np.dot(Z1_0e,best)
            Bout =  ( LHS - pay0est)/( pay1est - pay0est)
            #Bound beliefs in (0,1)
            Bout[Bout<0.001] = 0.001
            Bout[Bout>0.999] = 0.999
            crit = max(np.abs(CP1-CP1_out))
            Bin = Bout
            CP1 = CP1_out
            k0 = k0+1
        else:
            crit=0
    #End while (crit>tol)& (k0<=1)
    return([best, Bout, CP1_out, convergence])
