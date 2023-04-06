# Data-processing scripts

Scripts mainly concerned with sifting through, processing and aggregating the .csv-files. Data has been primarily partitioned into **climate**, i.e. sensor data and additional macroclimate variabels from SMHI, and **control**, i.e. setpoints, airflows and actuator states. Data is then merged into one MultiIndexed dataframe in data_merge.py.

## Notes

### Removed dates

There are several days in the dataset for which there are missing values for different sensors and/or setpoint settings, especially towards the end of December. These dates have been removed and accounted for and are specified in *misc/specs.json*.



