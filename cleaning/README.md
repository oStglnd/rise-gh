# Data cleaning

There are anomalies/outliers in the data that need to be accounted for. For example, we have missing data for certain dates, most notably around Christmas 2022, which is accounted for in the data loading. But there are also more subtle anomalies and structural features to account for, such as the changes in the temperature setpoint. 

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/setpoint.png?raw=true width="500" title="setpoint">
</p>

We can deal with setpoints anomalies manually. That is, we consider the normalized difference in real temperature and temperature setpoint rather than the actual measured temperature. However, there are other outliers which may need to be dealth with, for which we employ an Autoencoder. 

## Anomaly detection w. AE

Using an Autoencoder with convolutional layers, we can attempt to reconstruct segments of the time series. With sufficient confidence in the model's ability to properly reconstruct segments of the series from a latent space, we can use the difference in real and reconstructed values to flag anomalies. Training on data from December and January, we attempt to reconstruct data from February.

### Temperature measurements
The reconstructed temperature segments look pretty good and there are no segments in the test data for which the MAE comes close to anything in the training data. No anomalies detected when comparing maximum training MAE to reconstruction MAE for test data. For the first and last segment of the temperature series in February (whitened and removed temperature setpoints effect), we get the following reconstructions using a 120-60-Z-60-120 1D-conv AE:

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/temp_reconstructed_1.png?raw=true width="300" title="reconstruction">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/temp_reconstructed_2.png?raw=true width="300" title="reconstruction">
</p>


### Pressure measurements

Training an autoencoder with identical architecture, i.e. 128-64-Z-64-128, we obtain a similar result for the pressure measurements, where the data can be reconstructed well when trained on data from December and January, and then tested on data in February. The first and last segment of the pressure data are plotted below:

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/pressure_reconstructed_1.png?raw=true width="300" title="reconstruction">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/pressure_reconstructed_2.png?raw=true width="300" title="reconstruction">
</p>
