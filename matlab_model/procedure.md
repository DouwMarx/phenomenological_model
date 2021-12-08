* Define a speed profile in time domain. fr_time = f(t)  [Hz,rev/s] 
  * Convert the speed profile to angle domain angle(t) = integrate over time 
* Find the angles at which impulses will occur for all angles traversed
  * Do this by adding noise to the average angle between pulses.
  * Then, accumulate these distances to find the angles at which impulses will occur
  * accumulate until reaching the maximum angle traversed in a measurement
* Now, create an interpolation mapping angle -> time
  * Do this using the time vector and the angle vector from integration
* Use the interpolation mapping the find the exact (float) time points where impulses occur
  * Using the master sample frequency, set the indexes closest to the time points at which impulses to 1, otherwise zero.
* Compute the SDOF response using master sample frequency
  * Convolve response with the impulse signal
* Subsample to get a measurement.