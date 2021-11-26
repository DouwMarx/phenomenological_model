function [t,x,xNoise,frTime] = bearingSignalModelDist(d,D,contactAngle,n,faultType,fc,fd,fm,fr,N,...
         fs,SNR_dB,qFault,qStiffness,qRotation)
%% Generation of a simulated signal for distributed fault in rolling element bearing
%
% Input:
% d = bearing roller diameter [mm]
% D = pitch circle diameter [mm]
% contactAngle = contact angle [rad]
% n = number of rolling elements
% faultType = fault type selection: inner, outer, ball [string]
% fr = row vector containing the rotation frequency profile
% fc = row vector containing the carrier component of the speed
% fm = row vector containing the modulation frequency
% fd = row vector containing the frequency deviation
% N = number of points per revolution
% SNR_dB = signal to noise ratio [dB]
% qFault = amplitude modulation at the fault frequency
% qStiffness = amplitude value of the deterministic 
% component related to the stiffnessvariation
% qRotation = amplitude value of the deterministic component related to the bearing rotation
%
% Output:
% t = time signal [s]
% x = simulated bearing signal without noise
% xNoise = simulated bearing signal with noise
% frTime = speed profile in the time domain [Hz]
%
% G. D'Elia and M. Cocconcelli

switch faultType
    case 'inner'
        geometryParameter = 1 / 2 * (1 + d/D*cos(contactAngle)); % inner race fault
    case 'outer'
        geometryParameter = 1 / 2 * (1 - d/D*cos(contactAngle)); % outer race fault
    case 'ball'
        geometryParameter = 1 / (2*n) * (1 - (d/D*cos(contactAngle))^2)/(d/D); % outer racefault
end

Ltheta = length(fr);
theta = (0:Ltheta-1)*2*pi/N;
thetaTime = zeros(1,length(fr));
for index = 2:length(fr),
    thetaTime(index) = thetaTime(index - 1) + (2*pi/N)/(2*pi*fr(index));
end

L = floor(thetaTime(end)*fs); % signal length
t = (0:L-1)/fs;
frTime = interp1(thetaTime,fr,t,'spline');

% generating rotation frequency component
xRotation = qRotation * cos(fc/fc.*theta + fd./fc.*(cumsum(cos(fm./fc.*theta)/N)));
xRotationTime = interp1(thetaTime,xRotation,t,'spline');

% generating stiffness variation
tauStiffness = n / 2 * (1 - d/D*cos(contactAngle));
xStiffness = qStiffness * cos(fc./fc*tauStiffness.*theta + fd./fc*tauStiffness.*...
    (cumsum(cos(fm./fc*tauStiffness.*theta)/N)));
xStiffnessTime = interp1(thetaTime,xStiffness,t,'spline');

% amplitude modulation
tauFautl = n*geometryParameter;
q = 1 + qFault * sin(fc./fc*tauFautl.*theta + fd./fc*geometryParameter.*...
    (cumsum(cos(fm./fc*geometryParameter.*theta)/N)));
qTime = interp1(thetaTime,q,t,'spline');
xFaultTime = randn(1,L);
xFaultTime = xFaultTime .* qTime;

% adding therms
x = xFaultTime + xStiffnessTime + xRotationTime;

% Adding noise with given SNR
rng('default'); %set the random generator seed to default (for comparison only)
SNR = 10^(SNR_dB/10); %SNR to linear scale
Esym=sum(abs(x).^2)/(L); %Calculate actual symbol energy
N0 = Esym/SNR; %Find the noise spectral density
noiseSigma = sqrt(N0); %Standard deviation for AWGN Noise when x is real
nt = noiseSigma*randn(1,L);%computed noise
xNoise = x + nt; %received signal