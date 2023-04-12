# Models

Tested LSTM w. Convolutional embeddings and BatchNorm for both temperature and humidity. Seems to work very, very well. Correlation 0.98-0.999 for predictions and actual values. Should be enough to build environment.

	- Test also LSTM with ATTENTION layer, Seq2seq (i.e. RepeatVector and TimeDistributed)
	- (X) Seems "choice" of data for training is very important. Perhaps more feature engineering necessary? Or just more data. For example exploding validation loss and test data errors f. CNN model. Due to certain data points? Do another "sweep" of dataset...
	- (1) Create GMM models for flow and DC temp
	- (2) Read up on LSTMs & Attention, implement LSTM network "correctly". 
	- (3) Build GYM! For RL agent. Simple agent.

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/humid_preds_10min.png?raw=true width="300" title="Humidity, preds">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/humid_error_dist.png?raw=true width="300" title="Humidity, errors">
</p>

<p align="center">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/temp_preds_10min.png?raw=true width="300" title="Temp, preds">
  <img src=https://github.com/oStglnd/rise-gh/blob/main/misc/plots/temp_error_dist.png?raw=true width="300" title="Temp, errors">
</p>

## Notes on temperature modelling (OLD)

 - Looking a short-term normalized series, it seems as if though GH temperature is heavily influenced by outdoors temperature, sun GSI & volume, and the DC temp - as well as the air flow when flow is not zero/low variance (i.e. constant-ish).

 - W.r.t. size of GH and avg. size of flows, should be reasonable to consider temperature some 10 mins into the future. The QUESTION is how to properly estimate effects w.r.t. normalizing series. Perhaps something like

	- sequence-specific normalization? I.e. a sort of batch-normalization PRE training.
	- SEPARATE sequences based on some criterion like GSI > 0 or variance of flow is low. 

	- Use relative differences in explanatory variables to predict relative change in GH temp? 
	- Also try using absolute quantitites to predict absolute quantity.
	- Probs need to include "absolute" temperature anyway in order to correctly predict effect.
	- As well as DIFFERENCE between absolute temp in GH and absolute temp in DC, as well as outside. 

## Notes on data preprocessing

 - Due to changes in the *temperatures* setpoint, looking mainly at "corrected" temperature series where difference in setpoint has been account for, e.g. looking at the difference in temperature w.r.t. the "current" setpoint.

 - *Flow* variable for fan into GH has some seemingly erroneous values. Series has been clipped to only allow for "minimum" values at some 1700-1800 which seems to be "default" value when fan is turned off or turned down.

 - *DC temperature* is highly volatile and fluctuating in the short term. Hence, possibly using FFT-smoothed series instead.

 - For SMHI variables, used interpolation for create smooth series over minutes instead of only hourly measurements.

 - 



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