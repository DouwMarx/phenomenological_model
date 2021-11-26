* Define a speed profile in time domain. fr_time = f(t)  [Hz,rev/s] (row vector)
  * Convert the speed profile to angle domain angle(t) = integrate over time (fr_time(t)*2*pi)
* Find the angles at which impulses will occur for all angles traversed in a single sample
  * Do this by adding noise to the average angle between pulses.
* Then, cumulate these distances to find the angles at which impulses will occur
  * Cumulate until reaching the maximum angle traversed in a measurement
* Now, create an interpolation mapping that does angle -> time
  * Do this using the time vector and the angle vector from integration
  * Find the time points at which impulses will occur.
* Use the interpolation mapping the find the exact time points where impulses occur
  * Using the master sample frequency, set the indexes closest to the time points at which impulses to 1, otherwise zero.
* Compute the SDOF response using master sample frequency
  * Convolve it with the impulse map
* Subsample to get a measurement.