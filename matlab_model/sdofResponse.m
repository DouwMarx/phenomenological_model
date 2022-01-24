function [sdofRespTime] = sdofResponse(fs,k,zita,fn,Lsdof)
%% Acceleration of a SDOF system
% [sdofRespTime] = sdofResponse(fs,k,zita,fn,Lsdof)
%
% Input:
% fs = sample frequency [Hz]
% k = spring stiffness [N/m]
% zita = damping coefficient
% fn = Natural frequency [Hz]
% Lsdof = desired signal length [points]
%
% Output:
% sdofRespTime = acceleration (row vector)
%
% G. D�Elia and M. Cocconcelli

m = k/(2*pi*fn)^2;
F = 1;
A = F/m;
omegan = 2*pi*fn;
omegad = omegan*sqrt(1-zita^2);

t = (0:Lsdof-1)/fs;
% system responce
xt = A/omegad * exp(-zita*omegan*t).*sin(omegad*t); % displacement
xd = [0 diff(xt)*fs]; % velocity
sdofRespTime = [0 diff(xd)*fs]; % acceleration

# I am not 100% convinded that the numerical differentiation of the time reponse is a good idea. Exponential decay makes integration behave stragely.
# Will rather try analytical