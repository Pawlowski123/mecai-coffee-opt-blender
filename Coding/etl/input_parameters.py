import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import os
import logging

def recipe_def():
    """ Reading dataframe with coffee sample metadata and getting random recipe to be the model goal

    Parameters
    ----------
    df_norm_cost: pandas dataframe with coffee sample metadata obtained from create_cost_column function
    lot_limit: number of rows it should be obtained from original dataframe to consider as lot input

    Returns:
    ----------
    df_recipe: Pandas dataframe with one row, the model metadata reference, LOT_AVAILABILITY_KG = -1, it will be simulated later
    df_norm: Pandas dataframe with df_recipe row removed
    """

    logging.warning(os.getcwd())
    df_norm = pd.read_csv('Datasets/etl/cleaned_cqi_file.csv', sep = ';')
    logging.warning('File read successfully')
    logging.warning('.')

    # Creating df_recipe dataframe, it will be used as the model goal on the parameters
    recipe_random = list(np.random.randint(low = 0, high = len(df_norm), size = 1))
    df_recipe = df_norm.iloc[recipe_random]
    df_norm = df_norm.drop(axis = 0, index = recipe_random)
    logging.warning('Recipe dataframe and df_norm with recipe row removed created successfully')
    df_recipe['LOT_AVAILABILITY_KG'] = -1

    return df_recipe, df_norm


def create_cost_column(df_norm, cost_min = 5, cost_max = 50):
    """ Reading normalized CSV and creating algorithm to get a random parameters recipe and random lot number

    Parameters
    ----------
    df_norm: pandas dataframe with recipe row removed from normalized csv data
    cost_min: random minimum cost in BRL/kg
    cost_max: random maximum cost in BRL/kg

    Returns:
    ----------
    Pandas dataframe with coffee sample metadata and random parametrized cost (BRL/kg) column
    """

    df_norm['COST_BRL_KG'] = np.random.uniform(cost_min, cost_max, size = len(df_norm))
    logging.warning('Random cost column created with range: %.0f and %.0f', cost_min, cost_max)
    logging.warning('.')

    return df_norm

def input_lots(df_norm_cost, lot_limit = 10, return_df_norm_wout_lots = False):
    """ Reading dataframe with coffee sample metadata and random parametrized cost (BRL/kg) and creating model-input lot availability

    Parameters
    ----------
    df_norm_cost: pandas dataframe with coffee sample metadata obtained from create_cost_column function
    lot_limit: number of rows it should be obtained from original dataframe to consider as lot input

    Returns:
    ----------
    df_lot_input: Pandas dataframe with coffee sample metadata with lot_limit number of rows
    df_norm: Original Pandas dataframe with df_lot_input rows removed
    """

    # Creating random list with size lot_limit as parameter and range inside the df_norm_cost number of rows
    rows_lot_limit = list(np.random.randint(low = 0, high = len(df_norm_cost), size = lot_limit))
    df_lots_available = df_norm_cost.iloc[rows_lot_limit]
    df_norm_wout_lots_available = df_norm_cost.drop(axis = 0, index = rows_lot_limit)
    logging.warning('Datasets with %.0f lots metadata and df_norm_cost removing input lot successfully created', lot_limit)

    if return_df_norm_wout_lots == True:
        return df_lots_available, df_norm_wout_lots_available
    else:
        return df_lots_available

def target_volume(df_recipe, df_lots_available, simulate_broken_volume = False, low_perc = 0.2, high_perc = 0.8, broken_volume_perc = 1.5):
    """ Reading df_recipe to re-calculate possible volume to be achieved on LOT_AVAILABILITY_KG field

    Parameters
    ----------
    df_recipe: Pandas dataframe with one row, the model metadata reference
    df_lots_available: Pandas dataframe with coffee sample metadata with lot_limit number of rows
    simulate_broken_volume: boolean variable to define, when set to True, if the volume should be higher than the input lots availability, default: False
    Returns:
    ----------
    df_lot_input: Pandas dataframe with coffee sample metadata with lot_limit number of rows
    df_norm: Original Pandas dataframe with df_lot_input rows removed
    """
    
    input_total_volume = int(df_lots_available['LOT_AVAILABILITY_KG'].sum())
    df_recipe = df_recipe.rename(columns = {'LOT_AVAILABILITY_KG': 'VOLUME_NEED'})
    logging.warning('Dataset with Broken volume set as %s, low_perc set as %.1f and high_perc set as %.1f was successfully created', simulate_broken_volume, low_perc, high_perc)
    
    if simulate_broken_volume == False:
        df_recipe['VOLUME_NEED'] = np.random.randint(low = low_perc*input_total_volume, high = high_perc*input_total_volume, size = 1)
    else:
        df_recipe['VOLUME_NEED'] = np.random.randint(low = input_total_volume, high = broken_volume_perc*input_total_volume, size = 1)

    return df_recipe

def model_files(df_recipe, df_lots_available):
    """ Exporting CSV df_recipe after volume calculation and df_lot_input to Datasets/model

    Parameters
    ----------
    df_recipe: Pandas dataframe with one row, the model metadata reference
    df_lots_available: Pandas dataframe with coffee sample metadata with lot_limit number of rows
    
    Returns:
    ----------
    Does not return, export CSV file
    """
    
    # Exporting CSV files which will be used as inputs to the model
    logging.warning('.')
    logging.warning('.')
    logging.warning('.')
    df_recipe.to_csv('Datasets/model/input_recipe.csv', sep = ';')
    logging.warning('input_recipe.csv was successfully exported to Datasets/model/')
    df_lots_available.to_csv('Datasets/model/input_lots_available.csv', sep = ';')
    logging.warning('input_lots_available.csv was successfully exported to Datasets/model/')