# Overall Procedure for generating data: phenomenological model
  * Define a speed profile in time domain. fr_time = f(t)  [Hz,rev/s] 
    * Convert the speed profile to angle domain angle(t). Do this by integrating the speed profile over time. 
  * Find the angles at which impulses (angle dependent) will occur for all angles traversed.
    * Do this by adding noise to the average angle between pulses (slip/jitter).
    * Then, accumulate these noisy distances to find the angles of rotation at which impulses will occur
    * Accumulate these distances until reaching the maximum angle traversed (based on speed profile) in a measurement.
      Since the cumulative distance is stochatic, there is some uncertainty about exactly how many impulses will occur.
  * Now, create an interpolation mapping between angle -> time
    * Do this using the time vector and the angle vector obtained from integration
  * Use the interpolation mapping the find the exact (float) time points where impulses occur (We have previously computed the angles).
    * Using the master sample frequency, set the indexes closest to the time points at which impulses to 1, otherwise zero.
  * Compute the SDOF response using master sample frequency
    * Convolve response with the impulse signal
  * Subsample to get a measurement.

# Other notes on the implementation
* Work with a master sampling rate that represent continuous time (for instance 2*sampling frequency).s
 * This means that transients will not always take the same value at for instance the second sample of the transient.
