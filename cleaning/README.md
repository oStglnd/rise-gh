# Data cleaning

There are anomalies/outliers in the data that need to be accounted for. For example, we have missing data for certain dates, most notably around Christmas 2022, which is accounted for in the data loading. But there are also more subtle anomalies and structural features to account for, such as the changes in the temperature setpoint. 

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/setpoint.png?raw=true width="500" title="setpoint">
</p>

We can deal with setpoints anomalies manually. That is, we consider the normalized difference in real temperature and temperature setpoint rather than the actual measured temperature. However, there are other outliers which may need to be dealth with, for which we employ an Autoencoder. 

## Anomaly detection w. AE

Using an Autoencoder with convolutional layers, we can attempt to reconstruct segments of the time series. With sufficient confidence in the model's ability to properly reconstruct segments of the series from a latent space, we can use the difference in real and reconstructed values to flag anomalies. Training on data from December and January, we attempt to reconstruct data from February.

### Temperature measurements
The reconstructed temperature segments look pretty good and there are no segments in the test data for which the MAE comes close to anything in the training data. No anomalies detected when comparing maximum training MAE to reconstruction MAE for test data. For the first and last segment of the temperature series in February (whitened and removed temperature setpoints effect), we get the following reconstructions using a 120-60-Z-60-120 1D-conv AE:

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/temp_reconstructed_1.png?raw=true width="300" title="reconstruction">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/temp_reconstructed_2.png?raw=true width="300" title="reconstruction">
</p>


### Pressure measurements

Training an autoencoder with identical architecture, i.e. 128-64-Z-64-128, we obtain a similar result for the pressure measurements, where the data can be reconstructed well when trained on data from December and January, and then tested on data in February. There seems to be no significant outliers or anomalies in the testing data. The first and last segment of the pressure data are plotted below:

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/press_reconstructed_1.png?raw=true width="300" title="reconstruction">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/press_reconstructed_2.png?raw=true width="300" title="reconstruction">
</p>

## Denoising humidity measurements

There is a periodical noise component to the humidity measurements in the GH, due to a humidifying system being activated something like every $5$ minutes. In order to accurately assess the effect of other variables on indoors humidity, we would like to filter out the effect of the humidifying system on the measurements.

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/humidity_1d.png?raw=true width="500" title="reconstruction">
</p>

### Moving average

Since we know that the humidifier is activated every $5$ minutes, we might attempt to use a moving average. First, we consider the sensor average for each timestep, i.e. TA01_GT10X_GM10X. Second, since we have $2$ measurements every minute, we consider the $10$-step rolling averasge. The results are plotted below for a $6$ hour period.:

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/rolling_humidity_6h.png?raw=true width="500" title="reconstruction">
</p>

### FFT

We can also attempt to filter out the oscillations using the FFT algorithm. This achieves a very smooth result. However, 10-step moving avg. seems more straightforward for smoothing the humidity measurements.

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/fft_humidity_6h.png?raw=true width="500" title="reconstruction">
</p>

### Smoothed, normalized measurements

Normalizing the humidity measurements and applying the smoothing to the sensors avgs., we can see the relationships between DC humidity (DC_GT401_GM401), outdoors humidity (SMHI), and the sensors avgs. in the GH (TA01_GT10X_GM10X).

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/norm_humid_fft_5d.png?raw=true width="400" title="reconstruction">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/norm_humid_rolling_5d.png?raw=true width="400" title="reconstruction">
</p>