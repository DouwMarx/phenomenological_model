% Simulated localized and distributed fault in rolling
% element bearing
%
% G. D'Elia and M. Cocconcelli
clear;
clc;
close all
%% Bearing geometry
d = 8.4; % bearing roller diameter [mm]
D = 71.5; % pitch circle diameter [mm]
n = 16; % number of rolling elements
contactAngle = 15.7*pi/180; % contact angle
faultType = 'ball';

%% Speed profile
N = 500; % number of points per revolution (Discretize 1 revolution in so many points for interpolation sake?)
Ltheta = 2*N; % signal length (
theta = (0:Ltheta-1)*2*pi/N;
fc = 50;
fd = 0;
fm = 0.1*fc;
fr = fc + 2*pi*fd.*(cumsum(cos(fm.*theta)/N)); # fr is therefore defined as a angle dependent function?

%% Localized fault
varianceFactor = 0.04;
fs = 20480; % sample frequency [Hz]
k = 2e13;
zita = 0.02;
fn = 4230; % natural frequency [Hz]
Lsdof = 2^8;
SNR_dB = 3;
qAmpMod = 0.3;
[tLocal,xLocal,xNoiseLocal,frTimeLocal,meanDeltaTLocal,varDeltaTLocal,meanDeltaTimpOverLocal,...
    varDeltaTimpOverLocal,errorDeltaTimpLocal] = bearingSignalModelLocal(d,D,contactAngle,n,...
    faultType,fr,fc,fd,fm,N,varianceFactor,fs,k,zita,fn,Lsdof,SNR_dB,qAmpMod);
%figure;
%plot(tLocal(1:end),xLocal(1:end));
%xlabel('Time (s)','Fontname','Times New Roman');
%ylabel('Amplitude','Fontname','Times New Roman');
%title('Simulated Bearing fault Signal','Fontname','Times New Roman');
% grid on
% save SimuSignal.mat faultType fs SNR_dB tLocal xLocal xNoiseLocal 
%% Distributed fault
fs = 20480; % sample frequency [Hz]
SNR_dB = 0;
qFault = 1;
qStiffness = 0.1;
qRotation = 0.1;
% [tDist,xDist,xNoiseDist,frTimeDist] = bearingSignalModelDist(d,D,contactAngle,n,faultType,fc,fd,fm,...
%     fr,N,fs,SNR_dB,qFault,qStiffness,qRotation);
% figure;
% plot(tDist(1:2000),xNoiseDist(1:2000));
% xlabel('Time (s)','Fontname','Times New Roman');
% ylabel('Amplitude','Fontname','Times New Roman');
% title('Simulated Bearing fault Signal','Fontname','Times New Roman');
% grid on