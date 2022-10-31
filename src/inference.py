import argparse
import os
import sys
from glob import glob
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import ast

from pprint import pprint
from datetime import datetime, timedelta
from pickle import load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

from pytorch_lightning.callbacks import TQDMProgressBar
import torch
import dill


def generate_scaled_series(df, scaler, cat_cols, time_col, numerical_cols):
    '''
    Generates a Darts TimeSeries object and scales it according to the scaler object passed
    '''
    try:
        series = TimeSeries.from_group_dataframe(df,
                                            group_cols=cat_cols,
                                            time_col=time_col,
                                            value_cols=numerical_cols,
                                            freq='H')

        scaled_series = scaler.transform(series)
        return scaled_series

    except Exception as e:
        print(str(e))
        sys.exit(3)


def plot_predictions(target_column, target_series, pred_series, predictions_figure):
    '''
    Generates an image of the historical target series and the models' predictions for its' next 24 hours
    '''
    figsize = (9, 6)
    lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles" 

    # plot actual series
    plt.figure(figsize=figsize)
    target_series.plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title(f"Historical values and next 24h predictions for {target_column}")
    plt.savefig(predictions_figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path to the pretrained model")
    parser.add_argument("data_file", help="Path to the data used for inference")
    parser.add_argument("target_column", help="Name of the target variable")
    parser.add_argument("categorical_columns", help="Name of the categorical variables from the data file to be used for inference")
    parser.add_argument("numerical_columns", help="Name of the numerical variables from the data file to be used for inference")
    parser.add_argument("time_column", help="Name of the time variable in the data file")
    parser.add_argument("scaler_covariates", help="Path to the scaler fitted to the covariates")
    parser.add_argument("scaler_target", help="Path to the scaler fitted to the target")
    parser.add_argument("predictions_figure", help="Path to the image to dump the predictions")

    args = parser.parse_args()
    model = args.model
    data_file = args.data_file
    target_column = args.target_column
    categorical_columns = args.categorical_columns
    numerical_columns = args.numerical_columns    
    time_column = args.time_column
    scaler_covariates = args.scaler_covariates
    scaler_target = args.scaler_target
    predictions_figure = args.predictions_figure


    #Read the historical data which we will use for infering the next 24 hours
    try:
        df = pd.read_csv(data_file, parse_dates=time_column)
    except Exception as e:
        print(str(e))
        sys.exit(1)
    
    #Load the model
    try:
        my_model = torch.load(model, pickle_module=dill)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    #Load the covariates scaler
    try:
        scaler_covariates = load(open(scaler_covariates, 'rb'))
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    #Load the target scaler
    try:
        scaler_target = load(open(scaler_target, 'rb'))
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    #Load the categorical columns
    try:
        cat_cols = ast.literal_eval(categorical_columns)
    except Exception as e:
        print(str(e))
        sys.exit(2)

    #Load the numerical columns
    try:
        num_cols = ast.literal_eval(numerical_columns)
    except Exception as e:
        print(str(e))
        sys.exit(2)

    #Read the historical data which we will use for infering the next 24 hours and scale it according to the appropriate scaler
    covariates_scaled = generate_scaled_series(df, scaler_covariates, cat_cols, time_column, num_cols)
    target_scaled = generate_scaled_series(df, scaler_target, cat_cols, time_column, target_column)

    
    #Run the prediction for the next 24 hours
    try:
        pred_series = model.predict(n=24, num_samples=200, series=target_scaled, past_covariates=covariates_scaled)
    except Exception as e:
        print(str(e))
        sys.exit(4)


    #Plot the historical data and the predictions
    plot_predictions(target_column, target_scaled, pred_series, predictions_figure)

    return pred_series
    

if __name__ == "__main__":
    main()