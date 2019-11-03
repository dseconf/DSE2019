function f = identifyingmomentcondition(param, beta) 
%  Calculates the fixed point to eq (11) 

% The inverse matrix of the moment condition
invmat = eye(size(param.F0))-beta*param.F0; 

% Left hand side
lhs = param.logp(1)-param.logp(2); 

% Right hand side
rhs = beta*(param.F1(1,:) - param.F0(1,:) - ...
    param.F1(2,:) + param.F0(2,:))*(invmat\param.psi);

% Moment condition
f = lhs - rhs; 
