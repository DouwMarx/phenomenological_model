import matplotlib.pyplot as plt
import numpy as np

fs = 3800
freqrange = np.arange(0,int(fs/2))
fault_freq = 74

sine = 1+np.sin(freqrange*2*np.pi/fault_freq)
# logsine = np.log(sine)
sqsine = sine**2
quadsine = sine**4
sixsine = sine**6
expsine = np.exp(sine**2)

def normalize(signal):
    return signal/np.max(signal)

plt.figure()
for sig in [sine,sqsine,quadsine,expsine]:
    plt.plot(normalize(sig))


plt.legend(["sine","sqsine","cubesine","expsine"])
