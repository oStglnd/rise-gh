# Data cleaning

There are anomalies in the data that need to be accounted for. For example, we have missing data for certain dates, most notably around Christmas 2022, which is accounted for in the data loading. But there are also more subtle anomalies and structural features to account for, such as the changes in the temperature setpoint. 

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/setpoint.png?raw=true width="500" title="setpoint">
</p>

We can deal with setpoints anomalies manually. However, there are other outliers which need to be dealth with, for which we employ an Autoencoder. 

## Anomaly detection w. AE

Using an Autoencoder with convolutional layers, we can attempt to reconstruct segments of the time series. With sufficient confidence in the model's ability to properly reconstruct segments of the series from a latent space, we can use the difference in real and reconstructed values to flag anomalies. Training on data from December and January, we attempt to reconstruct data from February.

For the first segment of the temperature series in February (whitened and removed temperature setpoints effect), we get the following reconstruction using a 120-60-Z-60-120 1D-conv AE:

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/reconstructed_1.png?raw=true width="300" title="reconstruction">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/reconstructed_2.png?raw=true width="300" title="reconstruction">
</p>

