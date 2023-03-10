# Models

## PCA projections

On colder days, data is well separated in lower-dimensional subspace w.r.t. outdoors (damped) temperature and airflow volume from DC.

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/pca_20-2.png?raw=true width="300" title="PCA, 20/02">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/pca_21-2.png?raw=true width="300" title="PCA, 21/02">
</p>

On relatively warmer days, separation is increasingly more difficult w.r.t. outdoors (damped) temperature and airflow volume from DC.

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/pca_18-2.png?raw=true width="300" title="PCA, 18/02">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/pca_19-2.png?raw=true width="300" title="PCA, 19/02">
</p>


## GNN Test

## Background 


### Climate

### Control

Clear issues in maintaining temperature control, ranging $\pm 4^\circ$ C  around setpoint.

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/temp_setp_2h.png?raw=true width="300" title="Temperature, setpoint, 2 hours">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/temp_setp_10d.png?raw=true width="300" title="Temperature, setpoint, 10 days">
</p>

Similarly, strong fluctuations in overpressure (setpoint = 4 Pa).

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/pa_setp_2h.png?raw=true width="300" title="Pressure, setpoint, 2 hours">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/pa_setp_10d.png?raw=true width="300" title="Pressure, setpoint, 10 days">
</p>