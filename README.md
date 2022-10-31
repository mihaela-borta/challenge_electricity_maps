# My approach for solving the ML Engineer challenge

The model should predict 24 hours of `carbon_intensity_avg` according to multiple covariate variables and as well as its own past values. 

The idea is for the model to support several categorical columns such as `production mode` and `zone_name`.


## 1. Approach:

1.   Data:
> * check **missing data**. It might be that for some features the missing data might not be distributed uniformely at random, but in fact cover larger periods of time. This often happens when support for a new data stream is implemented -- before that point, we'll not see values for it.
> * check **missing target data**. Usually this data cannot be used in a straight forward way neither for training, nor evaluation.
> * check for other weird stuff like several data points for the same timestamp
> * **imputation of missing values**. Take the simplest approach and use the last variable. Probably a form of interpolation would be better, but it can be that when there is a strong seasonality this method is not adviceable.
> * **multicollinearity**. Try to correct for it as it can make estimates vague, imprecise and unreliable.

2.   Model:
> * the darts package was used as it supports various models (which one could evaluate more easily given more time)
> * a very simple maive baseline in which one simply outputs data for the past 24 hours is used
> * the more advancenced model used was Temporal Fusion Transformer due to the following:
>> * Support for temporal aspect
>> * Support for covariate variable(s) (time series, but also static)
>> * Support for quantile predictions (vs just point estimates)
>> * Some support for interpretability (though this was not explored in this case)
>> * The resulting data is comprised by 3 slices of multivariate time series. Due to the lack of time, we treat each of these separately (split each in 80/20 for train and validation). A MinMax scaler is used separately for each slice. 
>> * No hyperparam optimization is done due to lack of time

3. Evaluation:
> * MAPE is used for ease of interpretability, though it has issues: it's asymmetric and can yield null or infinite values. SMAPE or something similar might be better.
> * Only a single day worth of estimates is evaluated just for illustration. In a real scenario would be pretty much useless (insufficient at least).
> * Below is a comparison between the predictions of the naive and TFT model. (For the naive model the equivalent of 3 days of forecasts are made based on the last one, though it should only be that the last 3 days are used to forecast the next one.)

![Alt text](images/Naive_forecast.png?raw=true "Title") ![Alt text](images/TFT_forecast.png?raw=true "Title")

## 2. Productionalization

The TFT model was saved as a binary file -- this is used in the `src/inference.py` main file along side with the required data scalers in order to issue predictions.

The script returns 1 day of predictions (including percentile predictions).

Command to build the inference Docker image (on Windows 11):

`docker -H unix:///mnt/wsl/shared-docker/docker.sock build --tag em-docker .`

Command to create the container:

`docker -H unix:///mnt/wsl/shared-docker/docker.sock run -it  --entrypoint /bin/bash   em-docker`

Command to run the script:
`python src/inference.py --model_file="model/tft_model.pt" --data_file="data/train_data_0.csv" --target_column="carbon_intensity_avg" --categorical_columns="['zone_name', 'production_sources']" --numerical_columns="['total_export_avg', 'power_production_coal_avg', 'power_production_gas_avg', 'power_production_oil_avg', 'power_consumption_solar_avg', 'power_consumption_unknown_avg', 'carbon_origin_percent_nuclear_avg', 'carbon_origin_percent_wind_avg', 'carbon_origin_percent_geothermal_avg', 'carbon_origin_percent_hydro_discharge_avg', 'power_net_import_DE_avg', 'power_net_import_DK-DK1_avg', 'latest_forecasted_precipitation_avg', 'latest_forecasted_solar_avg', 'latest_forecasted_wind_x_avg', 'latest_forecasted_wind_y_avg']" --time_column="datetime" --scaler_covariates="model/scaler_cov_0.pkl" --scaler_target="model/scaler_target_0.pkl" --predictions_figure="predictions.png"`

Arguments for the script:
- `model_file` -- Path to the pretrained model
- `data_file` -- Path to the data used for inference
- `target_column` -- Name of the target variable
- `categorical_columns` -- Name of the categorical variables from the data file to be used for inference
- `numerical_columns` -- Name of the numerical variables from the data file to be used for inference
- `time_column` -- Name of the time variable in the data file
- `scaler_covariates` -- Path to the scaler fitted to the covariates
- `scaler_target` -- Path to the scaler fitted to the target
- `predictions_figure` -- Path to the image to dump the predictions



