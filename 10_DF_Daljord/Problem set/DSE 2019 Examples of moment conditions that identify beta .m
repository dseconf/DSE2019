%% Data lab on identification of discount factors in dynamic discrete choice models

% Daljord and Iskhakov Data lab. Gives applied examples from Daljord's
% lecture
%
% # Set the parameters of the problem
% # Solve for the value functions at the true primitives using full
% solution methods
% # Recover the discount factor using the moment conditions with no sampling variation
% # Examples of set identification
%
% Uses structures _param_ to keep true values of problem parameters and  primitives use _recovered_  to keep recovered primitives.


clc
clear

%% First use a stylized Zurcher type model 
% Set parameters of the problem. 

% # of states
param.J = 4; 


% # Costs
param.c = [0 0 1 2]'; 


% Discount factor
param.beta = 0.8; 


% Transitions: Upper triangular
param.F1 = [0.1 0.6 0.2; 
       0 0.1 0.6;
       0 0 0.2; 
       0 0 0]; 

param.F1 = [param.F1, 1-sum(param.F1,2)]; 



% Replacement decision resets the mileage
param.F0 = [1, 0, 0; 
    1, 0, 0; 
    1, 0, 0;
    1, 0, 0]; 

param.F0 = [param.F0, 1- sum(param.F0,2)];


fprintf('True cost function c \n')
param.c

fprintf('True beta \n')
param.beta

fprintf('Transitions F0 \n')
param.F0

fprintf('Transitions F0 \n')
param.F1



%% Solve for the value functions using NFXP

% Initialize the contraction mapping
crit = 1; 
v0 = zeros(param.J,1); 
v1 = zeros(param.J,1); 
itmax = 2000; 
it = 0; 

while max(max(crit)) > 10^(-13)
    v0old = v0; 
    v1old = v1; 
    v0 = param.beta*param.F0*log(exp(v0old) + exp(v1old));
    v1 = param.c + param.beta*param.F1*log(exp(v0old) + exp(v1old)); 
    crit = abs([v0-v0old, v1-v1old]); 
    if it > itmax
        disp('Failed to converge in %d iterations.\n', itmax)
        break
    end
    it = it + 1; 
end
param.v0 = v0; 
param.v1 = v1; 

fprintf('AVF at convergence in %d iterations. \n', it)
[param.v0, param.v1]


%% Construct the (true) choice probabilities from  the reduced form
param.p1 = exp(param.v1)./(exp(param.v0) + exp(param.v1)); 
param.p0 = 1-param.p1; 

fprintf('Reduced form choice probs p_0 and p_1. \n')
[param.p0, param.p1]

% Create data
param.psi = -log(1-param.p1); 
param.logp = log(param.p1./(1-param.p1));




%% Recover the discount factor

% * Compute the moment on a grid
% * Plot against beta
% * Recover beta as the zero of the moment 

% Set gridpoints
param.II = 1000; 
betagrid = linspace(0,0.99, param.II); 
momentofbeta = zeros(1,param.II); 
for ii = 1:param.II
    [momentofbeta(ii)]= identifyingmomentcondition(param, betagrid(ii)); 
end

% Plot
clf
hold on
plot(betagrid, momentofbeta)
plot(betagrid, zeros(1,param.II), 'k')
ylim([-1,1])
xlim([0,1])
ylabel('Moment condition')
xlabel('\beta')
title('Moment condition given p, q')
saveas(gcf,'PointIdentified.pdf')


% Set up optimizer and recover beta
options = optimset('Algorithm','trust-region-reflective', 'Display', 'iter-detailed', ...
    'TolFun', 10^(-12), 'TolX', 10^(-12));
startval = 0.5; 
recovered.beta = fminunc(@(beta) criterionfunction(param, beta), startval, options);
fprintf('Recovered beta \n')
recovered.beta


%% Invert out the cost function from the HM conditions given beta

rationalizingc = @(param, beta) param.logp - ...
    beta*(param.F1 - param.F0)*((eye(size(param.F0)) - beta*param.F0)\param.psi); 
recovered.c=rationalizingc(param, param.beta); 

fprintf('Unique c that rationalizes the data given true beta = %0.2f in the first column. True c in the second column. \n', param.beta)
[recovered.c param.c]



%% Rationalize the data for beta = 0
recovered.cmyopic=rationalizingc(param, 0);
fprintf('Unique c that rationalizes the data given beta = 0 in the first column. True c in the second column. \n')
[recovered.cmyopic param.c]



%% No discount factor that rationalizes the data

param.p1 = [0.3, 0.2, 0.7, 0.75]';
param.p0 = 1-param.p1; 

param.psi = -log(1-param.p1); 
param.logp = log(param.p1./(1-param.p1));

% Set gridpoints
param.II = 1000; 
betagrid = linspace(0,0.99, param.II); 
momentofbeta = zeros(1,param.II); 
for ii = 1:param.II
    [momentofbeta(ii)]= identifyingmomentcondition(param, betagrid(ii)); 
end

% Plot
clf
hold on
plot(betagrid, momentofbeta)
plot(betagrid, zeros(1,param.II), 'k')
ylim([-1,1])
xlim([0,1])
ylabel('Moment condition')
xlabel('\beta')
title('Moment condition given p, q')
saveas(gcf,'PointIdentified.pdf')



%% Set identification example: Moment condition has two solutions
% Not a replacement problem. 

% Simplify further to three states
param.J = 3; 

% Set transitions 
param.F1 = [0.3 0.2 ; 
       0 0.25 ;
       0.2 0.3]; 
   
param.F1 = [param.F1, 1- sum(param.F1,2)];

param.F0 = [0.9 0; 
    0 0.9; 
    0 1 ]; 

param.F0 = [param.F0, 1- sum(param.F0,2)];


% Set some arbitrary choice probabilities
param.p1 = [0.5, 0.49 0.2]'; 
param.p0 = 1-param.p1; 

% Create log choice probability ratios and corrected value functions
param.psi = -log(param.p0); 
param.logp = log(param.p1./param.p0);


%% Plot the moment condition over beta

% Calculate the moment conditions at a grid
param.II = 1000; 
betagrid = linspace(0,1-1/param.II, param.II); 
momentofbeta = zeros(param.II,1); 
for ii = 1:param.II
    [momentofbeta(ii)]= identifyingmomentcondition(param, betagrid(ii)); 
end

% Plot
clf
hold on
plot(betagrid, momentofbeta)
plot(betagrid, zeros(1,param.II), 'k')
ylim([-0.2,0.2])
xlim([0,1])
ylabel('Moment condition')
xlabel('\beta')
title('Moment condition given p, q. Set identified')
saveas(gcf,'SetIdentified.pdf')





%% Solve for all \beta that satisfy the moment condition
startvalprime = 1/3; 
startvalstar = 0.95; 
recovered.betaprime = fminunc(@(beta) criterionfunction(param, beta), startvalprime, options); 
recovered.betastar = fminunc(@(beta) criterionfunction(param, beta), startvalstar, options); 

fprintf('Recovered betas \n')
[recovered.betaprime recovered.betastar]


%% Invert the payoff function that rationalizes the data for each beta

% Define function that recovers the pay-offs that rationalize the choice
% probabilities given the discount factor
rationalizingc = @(param, beta) param.logp - ...
    beta*[param.F1 - param.F0]*[(eye(3) - beta*param.F0)\param.psi]; 


% Recover the costs for the two identified betas
recovered.cprime = rationalizingc(param, recovered.betaprime); 
recovered.cstar = rationalizingc(param, recovered.betastar); 

fprintf('Recovered cprime and cstar. \n')
[recovered.cprime recovered.cstar]



%%  Solve for the rationalizing costs for a range of beta values in [0,1). 
% For each beta, a unique vector of cees rationalizes the same data
betagrid = linspace(0,1-1/param.II,param.II); 
cees = zeros(param.J,param.II);

for ii = 1:param.II
    cees(:,ii)= rationalizingc(param, betagrid(ii)); 
end

% Plot the cees over beta
clf
hold on
plot(betagrid, cees(1,:))
plot(betagrid, cees(2,:))
plot(betagrid, cees(3,:))
ylim([-5/2,1/2])
xlim([0,1])
ylabel('c')
xlabel('\beta')
legend('c_1', 'c_2', 'c_3', 'location', 'Best')
title('Rationalizing cees')
saveas(gcf,'Rationalizingcs.pdf')

    
    


