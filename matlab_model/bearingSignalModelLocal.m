function [t,x,xNoise,frTime,meanDeltaT,varDeltaT,meanDeltaTimpOver,varDeltaTimpOver,errorDeltaTimp]...
    = bearingSignalModelLocal(d,D,contactAngle,n,faultType,fr,fc,fd,fm,N,varianceFactor,fs,k,...
    zita,fn,Lsdof,SNR_dB,qAmpMod)
%% Generation of a simulated signal for localized fault in rolling element bearing
%
% Input:
% d = bearing roller diameter [mm]
% D = pitch circle diameter [mm]
% contactAngle = contact angle [rad]
% n = number of rolling elements
% faultType = fault type selection: inner, outer, ball [string]
% fr = row vector containing the rotation frequency profile (should say if this is defined in angle domain or time???????!!!!
% fc = row vector containing the carrier component of the speed
% fm = row vector containing the modulation frequency
% fd = row vector containing the frequency deviation
% N = number of points per revolution
% varianceFactor = variance for the generation of the random contribution (ex. 0.04)
% fs = sample frequency of the time vector
% k = SDOF spring stiffness [N/m]
% zita = SDOF damping coefficient
% fn = SDOF natural frequency [Hz]
% Lsdof = length of the in number of points of the SDOF response
% SNR_dB = signal to noise ratio [dB]
% qAmpMod = amplitude modulation due to the load (ex. 0.3)
%
% Output:
% t = time signal [s]
% x = simulated bearing signal without noise
% xNoise = simulated bearing signal with noise
% frTime = speed profile in the time domain [Hz]
% meanDeltaT = theoretical mean of the inter-arrival times
% varDeltaT = theoretical variance of the inter-arrival times
% menDeltaTimpOver = real mean of the inter-arrival times
% varDeltaTimpOver = real variance of the inter-arrival times
% errorDeltaTimp = generated error in the inter-arrival times
%
% G. D'Elia and M. Cocconcelli

# Define the appropriate geometry parameter
if nargin < 14
    qAmpMod = 1;
end
switch faultType
    case 'inner'
        geometryParameter = 1 / 2 * (1 + d/D*cos(contactAngle)); % inner race fault
    case 'outer'
        geometryParameter = 1 / 2 * (1 - d/D*cos(contactAngle)); % outer race fault
    case 'ball'
        geometryParameter = 1 / (2*n) * (1 - (d/D*cos(contactAngle))^2)/(d/D); % outer racefault
end

# Get the average angular distance that should be rotated until an impulse occurs
Ltheta = length(fr); # The length of the speed profile vector which is given as a function of angle?
theta = (0:Ltheta-1)*2*pi/N # Discretisation for theta every angle that is traversed from start to finisha 0-> 2pi*nrevs: Technically specified by user
deltaThetaFault = 2*pi/(n*geometryParameter);  # The average angular distance that needs to be rotated before a fault occurs 1rev / (n_balls*geometry_param)

# Add noise to the distance traveled between impulses
numberOfImpulses = floor(theta(end)/deltaThetaFault);  # total_angular_distance_traveled / angular_distance_for_fault
meanDeltaTheta = deltaThetaFault;
varDeltaTheta = (varianceFactor*meanDeltaTheta)^2;
deltaThetaFault = sqrt(varDeltaTheta)*randn([1 numberOfImpulses-1]) + meanDeltaTheta; # Normal distribtuion over mean angular distance traveled between impulses

thetaFault = [0 cumsum(deltaThetaFault)]; # The angles at which a fault is expected
frThetaFault = interp1(theta,fr,thetaFault,'spline'); # Get the rotation frequencies at spedifically the angular positions where an impulse will occur

deltaTimp = deltaThetaFault ./ (2*pi*frThetaFault(2:end)); # Time it took to travel between impulses # This assumes that the speed remains constant between impulses?
tTimp = [0 cumsum(deltaTimp)]; # The time instances when impulses occur

L = floor(tTimp(end)*fs); % signal length The number of samples (time domain) until the final impulse in the time domain.
t = (0:L-1)/fs; % A time vector up until the final impulse [seconds]

frTime = interp1(tTimp,frThetaFault,t,'spline'); % The rotation frequency as a function of time (is this now how you would want to define it? for instance experimental data?

deltaTimpIndex = round(deltaTimp*fs); # The number of indexes (time domain) traveled between impulses
errorDeltaTimp = deltaTimpIndex/fs - deltaTimp; # The mistake you are making in assigning an impulse to a given discrete time step

indexImpulses = [1 cumsum(deltaTimpIndex)]; # Careful, using 1 based indexing: The indexes in time domain where impulses occur.
index = length(indexImpulses); # initialize index as the final index in which an impulse occurs.

while indexImpulses(index)/fs > t(end) # This is used to find the final impulse in the time domain
    index = index - 1;
end
indexImpulses = indexImpulses(1:index); # Use only the impulses that would occur before the end of the time period

# Compute some metrics as a sanity check
meanDeltaT = mean(deltaTimp); # Mean time between impulses # Returned from function
varDeltaT = var(deltaTimp);
meanDeltaTimpOver = mean(deltaTimpIndex/fs); # Teoretical variance of inter arival times? Empirical rather?
varDeltaTimpOver = var(deltaTimpIndex/fs);

x = zeros(1,L);
x(indexImpulses) = 1; # Set all indices where a impulse will occur equal to 1

% amplitude modulation
# My understanding of the amplitude modulation is that there is a angle dependent function that modifies the amplitude of the impulses in the impulse train.
if strcmp(faultType,'inner')
    if length(fc) > 1, # If the length of the carrier component of the speed is not a scalar:
        thetaTime = zeros(1,length(fr));
        for index = 2:length(fr),
            thetaTime(index) = thetaTime(index - 1) + (2*pi/N)/(2*pi*fr(index)); # This is essentially integration using eulers method?
        end
        fcTime = interp1(thetaTime,fc,t,'spline');
        fdTime = interp1(thetaTime,fd,t,'spline');
        fmTime = interp1(thetaTime,fm,t,'spline');
        q = 1 + qAmpMod * cos(2*pi*fcTime.*t + 2*pi*fdTime.*(cumsum(cos(2*pi*fmTime.*t)/fs)));
    else  #If the the carrier component of the speed is just a scalar. This means there is no load variation?
        q = 1 + qAmpMod * cos(2*pi*fc*t + 2*pi*fd*(cumsum(cos(2*pi*fm*t)/fs))); # 1+ since cos is -1 -> 1
    end
    x = q.*x;
end

[sdofRespTime] = sdofResponse(fs,k,zita,fn,Lsdof);
x = fftfilt(sdofRespTime,x); # This should give the same result as conv, but should be faster.

L = length(x);
%rng('default'); %set the random generator seed to default (for comparison only)
SNR = 10^(SNR_dB/10); %SNR to linear scale
Esym=sum(abs(x).^2)/(L); %Calculate actual symbol energy
N0 = Esym/SNR; %Find the noise spectral density
noiseSigma = sqrt(N0); %Standard deviation for AWGN Noise when x is real
nt = noiseSigma*randn(1,L);%computed noise
xNoise = x + nt; %received signal