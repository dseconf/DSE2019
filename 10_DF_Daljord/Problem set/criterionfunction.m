function S = criterionfunction(param, beta) 
% 	DSE 2019 constructs the criterion function. Calls
% 	identifyingmomentcondition.m. 

m = identifyingmomentcondition(param, beta); 

% Criterion function. Use L2 norm
S = m.^2; 

