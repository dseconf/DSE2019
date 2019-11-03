# -------------------------------------------------------------------
#       Replication of
#       Identification and Estimation of Dynamic Games
#       When Players Beliefs are Not in Equilibrium
#       Author:  Victor Aguirregabiria and Arvind Magesan
#
#       Replicated by: Jasmine Hao
#       First verion: May 1, 2019
# -------------------------------------------------------------------



# ---------------------------------------------------------------------
#
#  OUTLINE OF THE PROGRAM
#
#		Preamble: Programs to be used repeatedly in main program
#
#      1. MODEL
#          1.1. Parameters
#          1.2. Values of State Variables and State Space
#          1.3. Transition matrices of State Variables and States where game is in equilibrium
#          1.4. One-period payoff function
#
#      2. SOLUTION: Computing *Equilibrium* Choice probabilities. For each market type:
#			2.1  Procedure for obtaining equilibria that we will use repeatedly
#			2.2  Obtain MP Equilibrium CCPS for each player and mkt type
#
#------------------MONTE CARLO LOOP STARTS HERE--------------------------
#
#      3. Data Generation
#
#      4. Test of Equilibrium Beliefs
#
#      5. Non Parametric Estinmation of:
#         a.) g_function
#         b.) beliefs
#         c.) payoff differences
#         d.) payoffs
#
#      6. Parametric Estimation of Beliefs and Payoffs Using NPL
#          6.1 Estimate Player 1's payoffs parametrically without imposing equilibrium restrictions
#          6.2 Estimate Player 1's payoffs parametrically without imposing equilibrium restrictions
#
#------------------MONTE CARLO LOOP ENDS HERE--------------------------
#
#      7. Monte Carlo Results
#          7.1 Test
#          7.2 Parametric Payoffs
#          7.3 Parametric Beliefs
#          7.4 Parametric CCPs
#---------------------------------------------------------------------
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
from scipy.stats import chisquare
from scipy.stats import chi2
if (os.name == 'posix'):
    path = "/Users/haoyu/Documents/GitHub/DrugStorePriceCollusion/AM2019"
else:
    path = "C:\\Users\\Jasmine\\Documents\\Github\\DrugStorePriceCollusion\\AM2019"
os.chdir(path)

seed = 5333799
tol = 1e-12

from misc import *
#-------------------------------
#		Preamble
#-------------------------------

# CLOGIT  -  Maximum Likelihood estimation of McFadden's Conditional Logit
#            Some parameters can be restricted
#            Optimization algorithm: Newton's method with analytical
#            gradient and hessian
#
# by Victor Aguirregabiria
#      Last version:  December, 2001
#
# Format      {best,varest} = clogit(ydum,x,restx,namesb)
#
# Input        ydum    - (nobs x 1) vector of observations of dependet variable
#                        Categorical variable with values: {1, 2, ..., nalt}
#
#              x       - (nobs x (k * nalt)) matrix of explanatory variables
#                        associated with unrestricted parameters.
#                        First k columns correspond to alternative 1, and so on
#
#              restx   - (nobs x nalt) vector of the sum of the explanatory
#                        variables whose parameters are restricted to be
#                        equal to 1.
#
#              namesb  - (k x 1) vector with names of parameters
#
#
#  Output      best    - (k x 1) vector with ML estimates.
#
#              varest  - (k x k) matrix with estimate of covariance matrix

#-------------------------------
# 		Test purpose
#       DGP: generate randome selection
#-------------------------------
npar = 4
nalt = 3
nobs = 3000
mean = np.random.rand(12)
tol = 1e-12

cov = np.random.rand(12,12)
cov = np.dot(cov,cov.T)

x = np.random.multivariate_normal(mean,cov,size=nobs)
restx = np.random.randn(nobs,3)
b = np.array([1,2,3,4])


x_tilde = np.zeros((nobs,3))
for j in range(3):
     xbuff = x[:,int(npar*j):int(npar*(j+1))]
     x_tilde[:,j] = np.dot(xbuff,b) + restx[:,j]
x_tilde = (x_tilde.T - x_tilde.max(1)).T
x_tilde = np.exp(x_tilde)
x_tilde = (x_tilde.T / x_tilde.sum(1)).T

ydum = np.zeros(nobs)
for i in range(nobs):
    ydum[i] = np.random.choice(3,p = x_tilde[i,:])

namesb = ['b_1','b_2','b_3','b_4']
clogit(ydum,x,restx,namesb,1)

#*******************************************************************

#********* Main Program Begins Here*****************************

#********************************************************************
# ------------------------------------------------------------------
#  1.      Model
# ------------------------------------------------------------------

# 1.1 Set structural parameters

# Player 1
theta_10 = 2.4
theta_11 = 0
theta_12 = 3.0
theta_13 = 0.5


# Player 2
theta_20 = 2.4
theta_21 = -1.0
theta_22 = 3.0
theta_23 = 0.5

#Discount factor
betad = 0.95

# Matrix of true payoff parameters, each column is a player

theta_0 = np.array([[theta_10,theta_11,theta_12,theta_13],[theta_20,theta_21,theta_22,theta_23]])

# "Bias" is the parameter that determines whether beliefs are in equilibrium or not
#  If "Bias"=1 then beliefs are in equilibrium.
#  To get results in paper for biased beliefs, set bias = 0.5
bias = 1


# 1.2  State Variables and State Space

#  Note:
#  'states' is the matrix with all the common knowledge
#  state variables.
#      Number of columns   = # of variables
#                          = {Yit-1, Yjt-1, Zjt }
#
#      Number of rows      = # of states = 2*2*numz
#         Rows are sorted, in this order, by Yit-1, Yjt-1, Zjt
#
num_z = 5
z_incre = 1
z_jt = np.linspace(-2,2,num=num_z)

Y_itm1 = np.array([0,1])

states =  my_meshgrid([Y_itm1,Y_itm1,z_jt])

Y_states = states[:,0:2]
numstate = states.shape[0]

# 1.3  Transition of State Variables

# Exogenous transition of z_it: we assume that z_it is i.i.d (wrt time) uniform in each market
zj_tran = (1/len(z_jt))*np.ones((len(z_jt), len(z_jt)))

#zj_tran = eye(rows(z_jt));  #To keep z constant over time within market
zj_tran_cdf = zj_tran.cumsum(1)

# Creating transitions for each player for each action
# and stacking them into an array.
# First two matrices are player 1 for action 0 and 1,
# second two are player 2.

excomp = np.kron(np.ones((4,4)),zj_tran)

ord = [2*2,states.shape[0],states.shape[0]]
own_full_tran = np.zeros(ord)

# Creating Player 1's matrices
# Jasmine comment: if player 1 is inactive at state, the transition matrix is 0
inact = states[:,0] == 0
own_full_tran[0,:,:] = inact * excomp

act = states[:,0] == 1
own_full_tran[1,:,:] = act * excomp

# Creating Player 2's matrices
inact = states[:,1] == 0
own_full_tran[2,:,:] = inact * excomp

act = states[:,1] == 1
own_full_tran[3,:,:] = act * excomp

# [TODO:] I'm not sure what this does
      #Exogenous transition matrices
y0_2 = ( 0**Y_states[:,1] ) *( 1** (1-Y_states[:,1] ))
ftran_00 = y0_2 * own_full_tran[0,:,:]
ftran_10 = y0_2 * own_full_tran[1,:,:]
y1_2 = ( 1**Y_states[:,1] ) *( 0** (1-Y_states[:,1] ))
ftran_01 = y1_2 * own_full_tran[0,:,:]
ftran_11 = y1_2 * own_full_tran[1,:,:]

#The final "parameter" we need to set is the states where the game is in equilibrium
#We assume the game is in a particular equilibrium (the one where player 1 enters with higher probability)
#   when either of the players' z variables are at their endpoints

z_min = z_jt[0]
#These are the endpoints of the z variable
z_max = z_jt[-1]

z_ex =  (states[:,2]==z_min) + (states[:,2]==z_max)
z_ex = z_ex*1
z_in = 1-z_ex
z_ex = np.where(z_ex==1)[0]
z_in = np.where(z_in==1)[0]
# z_ex = indexcat(z_ex, maxc(z_ex));
# z_in = indexcat(z_in, maxc(z_in));

beq = bias == 1
zt = states[:,2]
at_z_ex = (zt==z_min) | (zt==z_max)
at_z_ex = at_z_ex*1

lambda_1 = at_z_ex + bias*(1-at_z_ex)
lambda_2 = at_z_ex + bias*(1-at_z_ex)


#1.4 Construct True Payoff Function
true_pi = np.empty((states.shape[0],0))
zfun = np.empty((states.shape[0],0))

for i0 in range(2):
    othind = list(range(2))
    othind.pop(i0)
    theta_i0 = theta_0[i0,:]
    states_i0 = np.vstack((states[:,i0],states[:,2])).T
    pi_i0_0 = np.dot(np.vstack((np.ones(states.shape[0]),states_i0[:,1],np.zeros(states.shape[0]),states_i0[:,0])).T,theta_i0)
    pi_i0_1 = np.dot(np.vstack((np.ones(states.shape[0]),states_i0[:,1],-np.ones(states.shape[0]),states_i0[:,0])).T,theta_i0)
    z_i0_0 = np.vstack((np.ones(states.shape[0]),states_i0[:,1],np.zeros(states.shape[0]),states_i0[:,0])).T
    z_i0_1 = np.vstack((np.ones(states.shape[0]),states_i0[:,1],-np.ones(states.shape[0]),states_i0[:,0])).T
    true_pi = np.hstack([true_pi,np.vstack([pi_i0_0,pi_i0_1]).T])
    zfun = np.hstack([zfun,z_i0_0,z_i0_1])

profit1_0 = true_pi[:,0] #player 1's profit over all states when player 2 is out
profit1_1 = true_pi[:,1] #player 1's profit over all states when player 2 is in
profit2_0 = true_pi[:,2] #player 2's profit over all states when player 1 is out
profit2_1 = true_pi[:,3] #player 2's profit over all states when player 1 is in

Z1_0 = zfun[:,0:4] #player 1's Z states when player 2 is out
Z1_1 = zfun[:,4:8] #player 1's Z states when player 2 is in
Z2_0 = zfun[:,8:12] #player 2's Z states when player 1 is out
Z2_1 = zfun[:,12:16] #player 2's Z states when player 1 is in

# -----------------------------------------------------------------------------
#  2.      Solution: Obtaining the Markov Perfect Equilibrium CCPS of the infinite horizon game
#
# -----------------------------------------------------------------------------


# ------------------------------------------------------
#      2.1. Procedure for equilibrium mapping
# ------------------------------------------------------
def MPE_br(pin, eqm):
    # and returns the best response choice probabilities.
    P1 = pin[:,0]
    P2 = pin[:,1]
    #Create choice specific values for each player given flow profit and guess of choice probabilities
    if eqm ==1:
        B1 = P1 #Player 2's Belief  = Player 1's CCP if eqm==1
        B2 = P2 #Player 1's Belief= Player 2's CCP  if eqm==1
    else:
        B1 = lambda_2*P1 #Player 2's Belief = biased fn of Player 1's CCP  if eqm\=1
        B2 = lambda_1*P2 #Player 1's Belief = biased fn of Player 2's CCP  if eqm\=1

    ZP1 = np.dot(np.diag(B2),Z1_1) + np.dot(np.diag(1-B2),Z1_0)
    ZP2 = np.dot(np.diag(B1),Z2_1) + np.dot(np.diag(1-B1),Z2_0)

    e0_P1 = 0.5772156649 - np.log(1-P1)
    e1_P1 = 0.5772156649 - np.log(P1)
    e0_P2 = 0.5772156649 - np.log(1-P2)
    e1_P2 = 0.5772156649 - np.log(P2)

    zbarP1 = np.dot(np.diag(P1),ZP1)
    zbarP2 = np.dot(np.diag(P2),ZP2)

    ebarP1 = P1*e1_P1 + (1-P1)*e0_P1
    ebarP2 = P2*e1_P2 + (1-P2)*e0_P2

    FP_1_1 = np.dot(np.diag(B2),ftran_11) + np.dot(np.diag(1-B2),ftran_10)
    FP_1_0 = np.dot(np.diag(B2),ftran_01) + np.dot(np.diag(1-B2),ftran_00)

    FP_2_1 = np.dot(np.diag(B1),ftran_11) + np.dot(np.diag(1-B1),ftran_01)
    FP_2_0 = np.dot(np.diag(B1),ftran_10) + np.dot(np.diag(1-B1),ftran_00)

    FbarP_1 = np.dot(np.diag(P1),FP_1_1) + np.dot(np.diag(1-P1),FP_1_0)
    FbarP_2 = np.dot(np.diag(P2),FP_2_1) + np.dot(np.diag(1-P2),FP_2_0)

    W1Pz = np.dot(np.linalg.inv(np.eye(FbarP_1.shape[0])-betad*FbarP_1 ),zbarP1)
    W2Pz = np.dot(np.linalg.inv(np.eye(FbarP_2.shape[0])-betad*FbarP_2 ),zbarP2)

    W1Pe = np.dot(np.linalg.inv(np.eye(FbarP_1.shape[0])-betad*FbarP_1 ),ebarP1)
    W2Pe = np.dot(np.linalg.inv(np.eye(FbarP_2.shape[0])-betad*FbarP_2 ),ebarP2)
    v1P_1 = np.dot((ZP1 + betad*np.dot(FP_1_1,W1Pz)),theta_0[0,:]) + betad*np.dot(FP_1_1,W1Pe)
    v0P_1 = np.dot((np.zeros(zbarP1.shape) + betad * np.dot(FP_1_0,W1Pz)),theta_0[0,:]) + betad*np.dot(FP_1_0,W1Pe)

    v1P_2 = np.dot((ZP2 + betad*np.dot(FP_2_1,W2Pz)),theta_0[1,:]) + betad*np.dot(FP_2_1,W2Pe)
    v0P_2 = np.dot((np.zeros(zbarP2.shape) + betad * np.dot(FP_2_0,W2Pz)),theta_0[1,:]) + betad*np.dot(FP_2_0,W2Pe)

    P1out = np.exp(v1P_1)/(np.exp(v0P_1)+np.exp(v1P_1))
    P2out = np.exp(v1P_2)/(np.exp(v0P_2)+np.exp(v1P_2))
    pout =  np.vstack([P1out,P2out]).T
    return(pout)

# This procedure takes as inputs a guess of choice probabilities for the two players
# Test:
pin = np.random.rand(states.shape[0],2)
pout = MPE_br(pin,1)

def payoffs(B2_est, CP1_est):
    #NPL Loop begins here
    k0 = 1
    bin = B2_est
    bout = np.ones(numstate)
    pin = CP1_est
    pout = bout
    crit = max(np.abs(pin-pout))
    theta_0
    while (crit>tol)& (k0<=1):
        ZP1 = np.dot(np.diag(bin),Z1_1e) + np.dot(np.diag(1-bin),Z1_0e)
        e0_P1 = 0.5772156649 - np.log(1-pin)
        e1_P1 = 0.5772156649 - np.log(pin)
        zbarP1 = np.dot(np.diag(pin),ZP1)
        ebarP1 = pin*e1_P1 + (1-pin)*e0_P1

        FP_1_1 = np.dot(np.diag(bin),ftran_11) + np.dot(np.diag(1-bin),ftran_10)
        FP_1_0 = np.dot(np.diag(bin),ftran_01) + np.dot(np.diag(1-bin),ftran_00)
        FbarP_1 = np.dot(np.diag(pin),FP_1_1) + np.dot(np.diag(1-pin),FP_1_0)
        W1Pz = np.dot(np.linalg.inv(np.eye(FbarP_1.shape[0])-betad*FbarP_1 ),zbarP1)
        W1Pe =  np.dot(np.linalg.inv(np.eye(FbarP_1.shape[0])-betad*FbarP_1 ),ebarP1)
        ztilde1 = ZP1 + betad*np.dot(FP_1_1,W1Pz)
        ztilde0 = np.zeros(zbarP1.shape) + betad*np.dot(FP_1_0,W1Pz)
        etilde1 = betad*np.dot(FP_1_1,W1Pe)
        etilde0 = betad*np.dot(FP_1_0,W1Pe)

        #Assign ztildes and etildes to data
        x = np.hstack([ztilde0[state_ids.astype(int),:],ztilde1[state_ids.astype(int),:]])
        restx = np.vstack([etilde0[state_ids.astype(int)],etilde1[state_ids.astype(int)]]).T
        ydum = actions1
        # ydum = abs(ydum-2)
        namesb = ["alp1","del1","theta1"]
        [best, varest, convergence] = clogit(ydum,x,restx,namesb,0)
        if (convergence):
            #Update Belief Estimates
            kern = np.dot((ztilde1-ztilde0),best) + etilde1-etilde0
            pout = 1/(1+np.exp(-kern))
            ctilde = betad*np.dot(FP_1_1,np.dot(W1Pz,best))-betad*np.dot(FP_1_0,np.dot(W1Pz,best)) + etilde1-etilde0
            LHS = np.log(pout/(1-pout)) - ctilde
            pay1est = np.dot(Z1_1e,best)
            pay0est = np.dot(Z1_0e,best)
            bout =  ( LHS - pay0est)/( pay1est - pay0est)
            #Bound beliefs in (0,1)
            bout[bout<0.001] = 0.001
            bout[bout>0.999] = 0.999
            crit = max(np.abs(pin-pout))
            bin = bout
            pin = pout
            k0 = k0+1
        else:
            crit=0
    #End while (crit>tol)& (k0<=1)
    return([best, bout, pout, convergence])
# ------------------------------------------------------
#      2.2. Solve for equilibria
# ------------------------------------------------------

#Solve for beliefs in equilibrium CCPs
pin = 1e-4*np.ones((numstate,2))
pout = pin/2;
crit = np.abs(pin-pout).max()
iter = 1
while crit>=tol:
    pout = MPE_br(pin, 1)
    crit = np.abs(pin-pout).max()
    pin = pout
    iter = iter+1

peq_1 = pout[:,0]
peq_2 = pout[:,1]

#Solve for biased-beliefs in equilibrium CCPs
pin = 1e-4*np.ones((numstate,2))
pout = pin/2;

crit = np.abs(pin-pout).max()
iter = 1
while crit>=tol:
    pout = MPE_br(pin, 0)
    crit = np.abs(pin-pout).max()
    pin = pout
    iter = iter+1

pnoeq_1 = pout[:,0]
pnoeq_2 = pout[:,1]

#Players' beliefs in the DGP
if (beq):
    #Player 1
    beliefs_2 = peq_2
    #Player 2
    beliefs_1 = peq_1
    CP_true = np.vstack([peq_1,peq_2]).T
    Q1_true = np.log(peq_1)-np.log(1-peq_1)
else:
    #Player 1
    beliefs_2 = lambda_1 * pnoeq_2
    #Player 2
    beliefs_1 = lambda_2 * pnoeq_1
    CP_true = np.vstack([pnoeq_1,pnoeq_2]).T
    Q1_true = np.log(pnoeq_1)-np.log(1-pnoeq_1)

#Get true gfunction
g_true = np.zeros((numstate,2))

for z0 in range(4):
    eqmpts = z_ex[int(2*(z0)):int(2*(z0+1))]
    X = np.array([[beliefs_2[eqmpts[0]],(1-beliefs_2[eqmpts[0]])],[beliefs_2[eqmpts[1]],1-beliefs_2[eqmpts[1]]]])
    Y = Q1_true[eqmpts]
    thisg0 = np.linalg.solve(X,Y)
    g_index = [int(i) for i in np.linspace(eqmpts[0],eqmpts[0]+4,num=5)]
    g_true[g_index,:] = np.tile(thisg0,(5,1))

#MONTE CARLO SIMULATION LOOP STARTS HERE
nsim = 100
#Objects from Test:
#      Collect Unrestricted and Restricted CCP estimates
estCCP = []
rawestCCP =[]

#      Collect Likelihood Ratio test statistics
kept = 0
convergence = 0
lrt = []

#Objects from Payoff and Belief Estimation:

#      Collect Non-parametric payoff and Belief Estimates
np_gfun_est = []
np_belief_est = []
np_payoff_diff_est = []
np_payoff_est = []

#      Collect Parametric Payoff and Belief Estimates
p_payoff_est_no_eq = []
p_belief_est_no_eq = []
p_cp_est_no_eq = []

p_payoff_est_eq = []
p_belief_est_eq = []
p_cp_est_eq = []

for sim0 in range(nsim):
    print("Simulation No.",sim0+1)
    M = 500 #Number of markets
    Tdata = 5 #Number of Time periods in Data
    # -----------------------------------
    #  3.         Data Generation
    # -----------------------------------

    #Set initial conditions for each market:
    #We will assume that for t=1 (initial states):
    # states with (y_1t-1, y_2t-1) = (0,0) occur 25% of the time
    # states with (y_1t-1, y_2t-1) = (1,0) occur 25% of the time
    # states with (y_1t-1, y_2t-1) = (0,1) occur 25% of the time
    # states with (y_1t-1, y_2t-1) = (1,1) occur 25% of the time
    Iy_grid = np.ones(4)/4
    # Iy_grid = Iy_grid.cumsum()
    ym1 = np.array([[0,0],[1,0],[0,1],[1,1]])
    Iz_grid = (1/(len(z_jt)))*np.ones(len(z_jt))
    # Iz_grid = Iz_grid.cumsum()
    zm1 = z_jt
    Initial_states = np.zeros((0,3))

    for m0 in range(M):
        this_yij = np.random.choice(2**2,p=Iy_grid)
        this_yij = ym1[this_yij,:]
        this_zij = np.random.choice(len(zm1),p=Iz_grid)
        this_zij = zm1[this_zij]
        Initial_states = np.concatenate((Initial_states,np.array([[this_yij[0],this_yij[1],this_zij]]) ),axis=0 )

    State_Data_full = np.zeros((0,3))
    Action_Data_full = np.zeros((0,2))

    #Outer loop in markets: for each market we create a sequence of data T periods long
    for m0 in range(M):
        state_m1 = Initial_states[m0,:]
        #Generate Tdata periods of data
        state_data = state_m1
        action_data = np.zeros((0,2))
        for t0 in range(Tdata):
            #Find index of current state in state matrix
            state_m1i = indfind(state_m1, states)
               #Draw new z here
            zstate_m1i = state_m1[2]
            zstate_m1i = indfind(zstate_m1i, z_jt)
            new_z = np.random.choice(z_jt,p=zj_tran[zstate_m1i,:])
            #Draw new actions for players 1 and 2
            #CCP'S at current state
            CCP_state_m1 =  CP_true[state_m1i,:]
            #Player 1 action
            act1t0 = np.random.choice(2,p=[1-CCP_state_m1[0],CCP_state_m1[0]])

            #Player 2 action
            act2t0 = np.random.choice(2,p=[1-CCP_state_m1[1],CCP_state_m1[1]])
            tmrw_state = np.array([act1t0,act2t0,new_z])
            state_data = np.vstack([state_data,tmrw_state])
            action_data = np.vstack([action_data,np.array([act1t0,act2t0])])
            state_m1 = tmrw_state
        State_Data_full = np.vstack([State_Data_full,state_data[0:Tdata,:]])
        Action_Data_full = np.vstack([Action_Data_full,action_data])

    #Reshaping Data
    # Action_Data = reshape( Action_Data_full, M*Tdata, 2);   #First M rows are from year 1, the second M from year two,etc
    Action_Data = Action_Data_full
    State_Data  = State_Data_full

    #Assign State ID number to each state in the data (useful for estimation later)
    state_ids = np.zeros(State_Data.shape[0])
    #
    for ii in range(State_Data.shape[0]):
    # #Find corresponding state in the data
         state_ids[ii] = indfind(State_Data[ii],states)

        # [TODO]: This part is not finished
    # *********************************/
    # * 4. Test of Eqm Beliefs        */
    # *********************************/
    #4.1.1 Estimating CCPs - unrestricted - imposing the stationarity in the dgp
    #      Also collect summary data for each time period
    #Get state count
    Mdotki = np.array([sum(state_ids==i) for i in range(numstate)])
    Mdotk = Mdotki
    # Mdotk = sumc(Mdotki);
    #Get action count
    M1k_1 = np.array([sum(np.multiply(state_ids==i,Action_Data[:,0])) for i in range(numstate)])
    # M1k_1 = sumc(M1k_1);
    M0k_1 = np.array([sum(np.multiply(state_ids==i,1-Action_Data[:,0])) for i in range(numstate)])
    M1k_2 = np.array([sum(np.multiply(state_ids==i,Action_Data[:,1])) for i in range(numstate)])
    M0k_2 = np.array([sum(np.multiply(state_ids==i,1-Action_Data[:,1])) for i in range(numstate)])

    # Estimate CCPs using Raw frequency
    p_u = np.array([(M1k_1/Mdotk),(M1k_2/Mdotk)]).T
    # replace nan
    p_u[np.isnan(p_u)] = 0.5

    #Bound pu in (0,1)
    p_u[p_u>0.999] = 0.999
    p_u[p_u<0.001] = 0.001
    #CCPs to be used in the test
    Unrestricted_CCPest = p_u
    #we estimate a vector \theta for each value of Y_1
    # unique values of the variable Y_1
    uniques = [0,1]
    p1_r = np.zeros((0,int(numstate/2)))#we collect the restricted p estimates here
    p2_r = np.zeros((0,int(numstate/2)))
    keep = []

    for y0 in range(2):
        #Find the points in the state space with (Y_1 = y0)
        indy0 = states[:,0]==y0
        numk = indy0.sum()
        state_data_y0 = State_Data[:,0]==0
        ydata_y0 = Action_Data
        zdata_y0 = state_ids
        zdata_y0 = zdata_y0[state_data_y0]

        p_uhat = p_u[indy0,:]

        M_k = Mdotk[indy0]
        M1k_1_y0 = M1k_1[indy0]
        M0k_1_y0 = M0k_1[indy0]
        M1k_2_y0 = M1k_2[indy0]
        M0k_2_y0 = M0k_2[indy0]

        #This is a procedure to find the (\theta, \lambda) that  maximizes the constrained likelihood.
        # It takes as an input an initial guess of \lambda, say \lambda^0 and two points in space of excluded variable zstar and finds the MLE \theta^1.
        #It then takes \theta^1 as given and finds the associated maximizer \lambda^1. It continues in this fashion
        #to convergence.

        # Test:

        # def cloglike(lam0):
          # local it, itt, pthetad1, theta_in, theta_out, lam_out, thetastar_in, thetastar_out, lamstar_in, lamstar_out, crit, lam_crit, theta_crit,  llik2, gradtheta,
          #   gradltheta, ctheta, gradctheta, gradctheta11, gradctheta12, gradctheta1k, gradctheta21, gradctheta22, gradctheta2k, ptheta, thetastarlamstar, dmatrix, invd_grad, invertible, keep,
          #   numrest, z2, z1, nozstara, nozstarb, tolp, converged;
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
                llik2 = -llik2*np.eye(numstate)

                gradltheta = np.hstack([M_k/M,M_k/M])*( np.hstack([p_uhat[:,0],p_uhat[:,1]])-ptheta)

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
    #Calculate LRT
    lrt_sim0 = 2*sum(M1k_1*(np.log(p_u[:,0])-np.log(p1_r) )+ M0k_1 *(np.log(1-p_u[:,0])-np.log(1-p1_r) )) + 2*sum (M1k_2 *(np.log(p_u[:,1])-np.log(p2_r) )+ M0k_2 *(np.log(1-p_u[:,1])-np.log(1-p2_r) ))
    # if min(keep)==1:
    lrt.append(lrt_sim0)
        #mdotks = mdotks~mdotk;
    estCCP.append(np.vstack([p1_r,p2_r]).T)
    rawestCCP.append(p_u)
    kept= kept+min(keep)
    convergence = convergence+converge

    # -----------------------------------------------------------------------
    # 5.   Non Parametric Estimation of Payoffs and Beliefs
    # -----------------------------------------------------------------------
    #Estimate gfunctions and Beliefs for each period in the data
    np_gfun_est_sim0 = []
    np_belief_est_sim0 = []
    np_payoff_diff_est_sim0 = []
    np_payoff_est_sim0 = []
    #CCP estimates for period t0
    Pi0 =  p_u[:,0]
    Pj0 =  p_u[:,1]
    # Create player i q function for period t0
    q_it0 = np.log(Pi0/(1-Pi0))
    #Estimate gfunction of player i: Exploit the fact that Y_jt-1 and Z_jt are excluded from g
    eqmpts = z_ex[0:4]
    X = np.vstack([Pj0[eqmpts],(1-Pj0[eqmpts])]).T
    Y = q_it0[eqmpts]
    g01 = np.kron(np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y)), np.ones((int(numstate/2),1)))

    eqmpts = z_ex[4:8]
    X = np.vstack([Pj0[eqmpts],(1-Pj0[eqmpts])]).T
    Y = q_it0[eqmpts];
    g02 = np.kron(np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y)), np.ones((int(numstate/2),1)))
    np_gfun_est_sim0 = np.vstack([g01,g02])

    #Estimate Beliefs
    b_jt0 =  (q_it0 - np_gfun_est_sim0[:,1])/( np_gfun_est_sim0[:,0]- np_gfun_est_sim0[:,1])

    #Replace Beliefs at extreme poits with CCPs of player 2
    b_jt0[z_ex] = Pj0[z_ex]

    # #Bound beliefs in (0,1)
    b_jt0[b_jt0 > 0.999] = 0.999
    b_jt0[b_jt0 < 0.001] = 0.001
    np_belief_est_sim0 = b_jt0

    # #Estimate Payoff Differences using the fact that Y_it-1 enters in payoff but not in continuation values (conditional on Y_it)
    np_payoff_diff_est_sim0 = np_gfun_est_sim0[0:int(numstate/2),:]-np_gfun_est_sim0[int(numstate/2):numstate,:]

    # #Estimate New Entrant Payoffs exploiting knowledge about Incumbency payoffs
    np_payoff_est_sim0 = np.vstack([profit1_1[int(numstate/2):numstate],profit1_0[int(numstate/2):numstate]]).T + np_payoff_diff_est_sim0

    if len(np_gfun_est)==0:
        np_gfun_est= np_gfun_est_sim0
        np_belief_est = np.matrix(np_belief_est_sim0).T
        np_payoff_diff_est = np_payoff_diff_est_sim0
        np_payoff_est = np_payoff_est_sim0

    else:
        np_gfun_est= np.hstack([np_gfun_est,np_gfun_est_sim0])
        np_belief_est = np.hstack([np_belief_est,np.matrix(np_belief_est_sim0).T])
        np_payoff_diff_est = np.hstack([np_payoff_diff_est,np_payoff_diff_est_sim0])
        np_payoff_est = np.hstack([np_payoff_est,np_payoff_est_sim0])
    # -------------------------------------------------------------------
    #  6. Parametric Estimation of Payoffs and Beliefs using NPL
    # -------------------------------------------------------------------
    #Player 1 data needed for payoff estimation
    actions1 = Action_Data[:,0]
    Z1_1e = Z1_1[:,[0,2,3]]
    Z1_0e = Z1_0[:,[0,2,3]]

    #Procedure for estimating payoff parameters and beliefs
    #Takes as inputs player one beliefs about player 2 and player one ccps and returns parametric payoff and belief estimates


    # 6.1 Estimate Player 1's payoffs parametrically without imposing equilibrium restrictions
    #Initialize with non parametric estimates
    # Use same period of data for parametric estimation as for test
    B2_est = np_belief_est_sim0
    CP1_est = p_u[:,0]

    [payest,belest,pest,convergence] = payoffs(B2_est, CP1_est)
    if convergence==1:
        #Select belief and cp estimates for MC experiments
        sel = [2,7,12,17]
        if len(p_payoff_est_no_eq)==0:
            p_payoff_est_no_eq = np.matrix(payest).T
            p_belief_est_no_eq = np.matrix(belest[sel]).T
            p_cp_est_no_eq = np.matrix(pest[sel]).T
        else:
            p_payoff_est_no_eq = np.hstack([p_payoff_est_no_eq,np.matrix(payest).T])
            p_belief_est_no_eq = np.hstack([p_belief_est_no_eq,np.matrix(belest[sel]).T])
            p_cp_est_no_eq = np.hstack([p_cp_est_no_eq,np.matrix(pest[sel]).T])

    #Estimate Player 1's payoffs parametrically imposing equilibrium restrictions

    #Initialize with non parametric estimates

    # Use same period of data for parametric estimation as for test
    B2_est = p_u[:,1]
    CP1_est = p_u[:,0]

    [payest, belest, pest, convergence] = payoffs(B2_est, CP1_est)

    if len(p_belief_est_eq)==0:
        p_payoff_est_eq = np.matrix(payest).T
        p_belief_est_eq = np.matrix(belest[sel]).T
        p_cp_est_eq = np.matrix(pest[sel]).T
    else:
        p_payoff_est_eq = np.hstack([p_payoff_est_eq,np.matrix(payest).T])
        p_belief_est_eq = np.hstack([p_belief_est_eq,np.matrix(belest[sel]).T])
        p_cp_est_eq = np.hstack([p_cp_est_eq,np.matrix(pest[sel]).T])
#End for sim0 in range(nsim):
# -----------------------------------------------
#      7. RESULTS FROM MONTE CARLO EXPERIMENTS
# -----------------------------------------------

#7.1 The Test

pctgrid = [0.1,0.25,0.50,0.75,0.80,0.90,0.95,0.99]
ordLRT = np.sort(lrt)
# Quantiles of the test statistics and of the Chi-square under the null
qlrt = [np.quantile(ordLRT,pct) for pct in pctgrid]
q_chi= [chi2.ppf(xx,16) for xx in pctgrid]

print("")
print("   -------------------------------------------------------------------------------------")
print("       MONTE CARLO EXPERIMENT: QUANTILES OF THE STATISTICS")
print("   ------------------------------------------------------------------------------------")
print("       Percentile      Q-Chi(16)        Q-LRT     Q-PEARSON  ")
print("   -------------------------------------------------------------------------------------")
for j in range(len(pctgrid)):
    print(pctgrid[j],q_chi[j],qlrt[j])
print("   -------------------------------------------------------------------------------------")

# testtable = print(pctgrid,q_chi,qlrt)
# ret = xlsWrite(testtable, "results_25042017.xlsx", "i3", 1, 0);



# Empirical distribution of P-values
# plrt = cdfChinc(ordlrt, 16, 0);
# q_plrt = [np.quantile(plrt,1-pctgrid);

# print("   -------------------------------------------------------------------------------------")
# print("       MONTE CARLO EXPERIMENT: EMPIRICAL DISTRIBUTION OF PVALUES OF THE LRT STATISTIC")
# print("   -------------------------------------------------------------------------------------")
# print("       Percentile      Q-Pvalue-LRT ")
# print("   -------------------------------------------------------------------------------------")
# for j in range(len(pctgrid)):
#     print(1-pctgrid[j],q_plrt[j])
#
# print("   -------------------------------------------------------------------------------------")

# testtable = (1-pctgrid)~q_plrt;
# ret = xlsWrite(testtable, "results_25042017.xlsx", "m3", 1, 0);

maxalp = 0.15
minalp = 0.001
numalp= 1000
incalp = (maxalp-minalp)/(numalp-1)
alp_grid = np.linspace(minalp,maxalp,num=numalp)
ralp_grid = [chi2.ppf(1-xx,16) for xx in 1-alp_grid]

#Fraction of empirical test statistics that lie in rejection region given alpha
# simrejects = (ralp_grid.<=lrt');
# simrejectprob = sumr(simrejects)/rows(lrt);
# struct plotControl myPlot;
# myPlot = plotGetDefaults("xy");
# location = "top right";
# orientation = 0;
# label = "alpha"$|"Prob(rejection)";
# plotSetLegend(&myPlot, label, location, orientation);
# #plotSetYLabel(&myPlot, "", "verdana", 10, "black");
# plotSetXLabel(&myPlot, "alpha", "verdana", 10, "black");
# plotSetBkdColor(&myPlot, "white");
# plotxy(myPlot, alp_grid, alp_grid~simrejectprob ) ;


#7.2 Parametric Payoffs
#Calculate MAB and MSE of payoff estimates both under eqm restrictions and without
true_pay = [theta_10,theta_12,theta_13]
xtrue_pay = np.kron(true_pay,[1,0])
# ret = xlsWrite(xtrue_pay, "results_25042017.xlsx", "a3", 1, 0);

#Under Eqm Restrictions

MAB_pay_eq = np.abs(p_payoff_est_eq.T-true_pay).mean(0)
MAB_pay_eq = np.kron(MAB_pay_eq,[1,0])+ np.kron((MAB_pay_eq/true_pay),[0,1])
rMSE_pay_eq = np.sqrt(np.square(p_payoff_est_eq.T-true_pay).mean(0) )
rMSE_pay_eq= np.kron(rMSE_pay_eq,[1,0])+ np.kron(rMSE_pay_eq/true_pay,[0,1])
# ret = xlsWrite(MAB_pay_eq~rMSE_pay_eq, "results_25042017.xlsx", "c3", 1, 0);

#No Eqm Restrictions
MAB_pay_no_eq = np.abs(p_payoff_est_no_eq.T-true_pay).mean(0)
MAB_pay_no_eq = np.kron(MAB_pay_no_eq,[1,0])+ np.kron((MAB_pay_no_eq/true_pay),[0,1])
rMSE_pay_no_eq = np.sqrt(np.square(p_payoff_est_no_eq.T-true_pay).mean(0) )
rMSE_pay_no_eq= np.kron(rMSE_pay_no_eq,[1,0])+ np.kron(rMSE_pay_no_eq/true_pay,[0,1])
# ret = xlsWrite(MAB_pay_no_eq~rMSE_pay_no_eq, "results_25042017.xlsx", "f3", 1, 0);

print("")
print("   -------------------------------------------------------------------------------------")
print("       MONTE CARLO EXPERIMENT: ESTIMATES OF PAYOFF PARAMETERS")
print("   -------------------------------------------------------------------------------------")
print("       True             Equilibrium Restrictions           No Equilibrium Restrictions  ")
print("                           MAB         rMSE                     MAB         rMSE        ")
"   -------------------------------------------------------------------------------------"
print(np.vstack([xtrue_pay,MAB_pay_eq,rMSE_pay_eq,MAB_pay_no_eq,rMSE_pay_no_eq]))


"   -------------------------------------------------------------------------------------";
"-";

#7.3 Parametric Beliefs
#Calculate MAB and MSE of belief estimates both under eqm restrictions and without at several points
true_bel = beliefs_2[sel]
xtrue_bel = np.kron(true_bel,[1,0])

# ret = xlsWrite(xtrue_bel, "results_25042017.xlsx", "a12", 1, 0);

#Under Eqm Restrictions
MAB_bel_eq = np.abs(p_belief_est_eq.T-true_bel ).mean(0)
MAB_bel_eq = np.kron(MAB_bel_eq,[1,0])+np.kron((MAB_bel_eq/true_bel),[0,1])
rMSE_bel_eq = np.sqrt(np.square(p_belief_est_eq.T-true_bel).mean(0) )
rMSE_bel_eq = np.kron(rMSE_bel_eq,[1,0])+ np.kron(rMSE_bel_eq/true_bel,[0,1])
# ret = xlsWrite(MAB_bel_eq~rMSE_bel_eq, "results_25042017.xlsx", "c12", 1, 0);

#No Eqm Restrictions
MAB_bel_no_eq = np.abs(p_belief_est_no_eq.T-true_bel ).mean(0)
MAB_bel_no_eq = np.kron(MAB_bel_no_eq,[1,0])+np.kron((MAB_bel_no_eq/true_bel),[0,1])
rMSE_bel_no_eq =  np.sqrt(np.square(p_belief_est_no_eq.T-true_bel).mean(0) )
rMSE_bel_no_eq = np.kron(rMSE_bel_no_eq,[1,0])+ np.kron(rMSE_bel_no_eq/true_bel,[0,1])
# ret = xlsWrite(MAB_bel_no_eq~rMSE_bel_no_eq, "results_25042017.xlsx", "f12", 1, 0);


print("")
print("   -------------------------------------------------------------------------------------")
print("       MONTE CARLO EXPERIMENT: ESTIMATES OF BELIEF PARAMETERS")
print("   -------------------------------------------------------------------------------------")
print("       True             Equilibrium Restrictions           No Equilibrium Restrictions  ")
print("                           MAB         rMSE                     MAB         rMSE        ")
print("   -------------------------------------------------------------------------------------")
print(np.vstack([xtrue_bel,MAB_bel_eq,rMSE_bel_eq,MAB_bel_no_eq,rMSE_bel_no_eq]))

print(
"   -------------------------------------------------------------------------------------")
print("-")



#7.4 Parametric CCPs
#Calculate MAB and MSE of belief estimates both under eqm restrictions and without at several points
true_CP = CP_true[sel,1]
xtrue_CP = np.kron(true_CP,[1,0])
# ret = xlsWrite(xtrue_CP, "results_25042017.xlsx", "a24", 1, 0);

#Under Eqm Restrictions

MAB_CCP_eq = np.abs(p_cp_est_eq.T-true_CP ).mean(0)
MAB_CCP_eq = np.kron(MAB_CCP_eq,[1,0])+np.kron((MAB_CCP_eq/true_CP),[0,1])
rMSE_CCP_eq = np.sqrt(np.square(p_cp_est_eq.T-true_CP).mean(0) )
rMSE_CCP_eq = np.kron(rMSE_CCP_eq,[1,0])+ np.kron(rMSE_CCP_eq/true_CP,[0,1])
# ret = xlsWrite(MAB_CCP_eq~rMSE_CCP_eq, "results_25042017.xlsx", "c24", 1, 0);

#No Eqm Restrictions
MAB_CCP_no_eq = np.abs(p_cp_est_no_eq.T-true_CP ).mean(0)
MAB_CCP_no_eq = np.kron(MAB_CCP_no_eq,[1,0])+np.kron((MAB_CCP_no_eq/true_CP),[0,1])
rMSE_CCP_no_eq = np.sqrt(np.square(p_cp_est_no_eq.T-true_CP).mean(0) )
rMSE_CCP_no_eq = np.kron(rMSE_CCP_no_eq,[1,0])+ np.kron(rMSE_CCP_no_eq/true_CP,[0,1])
# ret = xlsWrite(MAB_CCP_no_eq~rMSE_CCP_no_eq, "results_25042017.xlsx", "f24", 1, 0);

print("")
print("   -------------------------------------------------------------------------------------")
print("       MONTE CARLO EXPERIMENT: ESTIMATES OF CCP PARAMETERS")
print("   -------------------------------------------------------------------------------------")
print("       True             Equilibrium Restrictions           No Equilibrium Restrictions  ")
print("                           MAB         rMSE                     MAB         rMSE        ")
print("   -------------------------------------------------------------------------------------")
print(np.vstack([xtrue_CP,MAB_CCP_eq,rMSE_CCP_eq,MAB_CCP_no_eq,rMSE_CCP_no_eq]))

print("   -------------------------------------------------------------------------------------")
print("-")
