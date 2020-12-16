import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import logging
import time

def model_input_read():
	""" Model CSV file inputs. Reading both recipe and available lots CSV files.
    
        Parameters
        ----------
        No parameters

        Returns:
        ----------
        df_recipe: dataframe, recipe reference
		df_available_lots: dataframe, available lots parameters, cost and availability
	"""

	# Changing reference directory to read CSV files from correct path
	os.chdir('C:/Users/BC966HL/OneDrive - EY/4. Mestrado/Dev/')
	os.getcwd()

	# Reading recipe reference and available lots
	df_recipe = pd.read_csv('Datasets/model/input_recipe.csv', sep = ';')
	df_available_lots = pd.read_csv('Datasets/model/input_lots_available.csv', sep = ';')
	logging.warning('CSV Files read successfully\n')
	logging.warning('.')

	return df_recipe, df_available_lots

def constraints_bounds(df_recipe, df_available_lots, ub = 1):
	""" Aroma, flavor, aftertaste, acidity, body, balance, moisture and recipe volume constraints created.
		Bounds in order to allocate the n-lot mass between 0 and the n-lot availability in kilograms.
    
        Parameters
        ----------
		df_recipe: dataframe containing a single row as recipe parameters
		df_available_lots: dataframe containing n rows, will be considered as the available lots with its own parameters to be worked out by the model
        ub: delta between recipe and allocation upper bound, will have the default as 1. The lower bound is 0.

        Returns:
        ----------
        constraints: list of constraints created on function
		bnd: matrix with lower and upper bounds per lot
	"""
	# Recipe constraints. This constraints are intended to make sure the delta between the parameter and the recipe is less than 1 in module, except for the moisture
	# which has lower bound 0 and upper bound 0.01.
	# Also creates a const_volume_need constraints to guarantee the recipe volume is achieved
	# This considers the tasting parameters to be linear, as a weighted average between lot allocation mass and the tasting parameter
	const_aroma = NonlinearConstraint(lambda x: abs(float(df_recipe['AROMA']) - (sum(np.array(df_available_lots['AROMA'])*x))/sum(x)), 0, ub)
	logging.warning('Aroma constraint created successfully\n')
	const_flavor = NonlinearConstraint(lambda x: abs(float(df_recipe['FLAVOR']) - (sum(np.array(df_available_lots['FLAVOR'])*x))/sum(x)), 0, ub)
	logging.warning('Flavor constraint created successfully\n')
	const_aftertaste = NonlinearConstraint(lambda x: abs(float(df_recipe['AFTERTASTE']) - (sum(np.array(df_available_lots['AFTERTASTE'])*x))/sum(x)), 0, ub)
	logging.warning('Aftertaste constraint created successfully\n')
	const_acidity = NonlinearConstraint(lambda x: abs(float(df_recipe['ACIDITY']) - (sum(np.array(df_available_lots['ACIDITY'])*x))/sum(x)), 0, ub)
	logging.warning('Acidity constraint created successfully\n')
	const_body = NonlinearConstraint(lambda x: abs(float(df_recipe['BODY']) - (sum(np.array(df_available_lots['BODY'])*x))/sum(x)), 0, ub)
	logging.warning('Body constraint created successfully\n')
	const_balance = NonlinearConstraint(lambda x: abs(float(df_recipe['BALANCE']) - (sum(np.array(df_available_lots['BALANCE'])*x))/sum(x)), 0, ub)
	logging.warning('Balance constraint created successfully\n')
	const_moisture = NonlinearConstraint(lambda x: abs(float(df_recipe['MOISTURE']) - (sum(np.array(df_available_lots['MOISTURE'])*x))/sum(x)), 0, .01)
	logging.warning('Moisture constraint created successfully\n')
	const_volume_need = NonlinearConstraint(lambda x: sum(x), .95*float(df_recipe['VOLUME_NEED']), 1.05*float(df_recipe['VOLUME_NEED']))
	logging.warning('Recipe volume constraint created successfully\n')

	# Creating bound matrix with shape 2 x n, as n is the number of lots available. Lower bound = 0 and upper bound = lot n availability
	# This will make sure the lot allocation mass will be positive (>= 0) and less than the lot availability 
	bnd = []
	for limit in df_available_lots['LOT_AVAILABILITY_KG']:
		bnd.append((0, int(limit)))

	constraints = [const_aroma, const_flavor, const_aftertaste, const_acidity, const_body, const_balance, const_moisture, const_volume_need]
	return constraints, bnd

def cost_obj_f(x, cost_per_lot):
	""" Objective function. The model will seek to minimize this objective function.
		It seeks to find the minimum production cost, achieving all constraints defined on constraints_bound function
    
        Parameters
        ----------
        x: allocation array, should not be passed as parameter on the minimize function
		cost_per_lot: df_available_lots['COST_BRL_KG']. The n-lot cost in BRL per kilogram, should be passed as args on the minimize function

        Returns:
        ----------
        The sum of production cost per lot with respective allocation mass in kilograms.
		It will return the square root of the objective function, since the algorithm behave better that way. It does not change the result, but
		make the round to fail less.
	"""
    
	return np.sqrt(sum(np.array(cost_per_lot)*x))


def minimize_round(df_recipe, df_available_lots, constraints, bnd, x0, maxiter = 10):
	""" Runs the SLSQP minimize algorithm with all defined constraints and bounds
        Parameters
        ----------
		df_recipe: dataframe containing a single row as recipe parameters
		df_available_lots: dataframe containing n rows, will be considered as the available lots with its own parameters to be worked out by the model
        constraints: list of constraints created on function
		bnd: matrix with lower and upper bounds per lot

        Returns:
        ----------
        slsqp: OptmizeResult module containing all minimize outputs
	"""
	# Running the Sequential Least SQuares Programming algorithm until it gets a successfull output.
	#slsqp.status = -1
	#while(slsqp.status != 0):
	x0 = np.random.randint(0, 100, len(df_available_lots['COST_BRL_KG']))
	t0 = time.perf_counter()
	slsqp = minimize(cost_obj_f,
					x0 = x0,
					args = (df_available_lots['COST_BRL_KG'],),
					method = 'SLSQP',
					constraints = constraints,
					bounds = bnd,
					options = {'maxiter': maxiter}
				)
	tf = time.perf_counter()
	round_time = tf - t0

	return slsqp, {'extra_data': {'round_time': round_time, 'x0': x0}}

def simulation_fixed_x0_var_iterations(df_recipe, df_available_lots, ub = 0.7, stop = 100, step = 10):
    simulation_fx_x0_var_iter = {}
    x0 = np.random.randint(0, df_available_lots['LOT_AVAILABILITY_KG'], len(df_available_lots['COST_BRL_KG']))
    constraints, bnd = constraints_bounds(df_recipe, df_available_lots, ub = 0.7)
    for iteration in range(1, stop, step):
        simulation_fx_x0_var_iter[str(iteration)] = minimize_round(df_recipe, df_available_lots, constraints, bnd, x0 = x0, maxiter = iteration)
        logging.warning('Appended iteration %d', iteration)
    return simulation_fx_x0_var_iter

def simulation_var_x0_fixed_iterations(df_recipe, df_available_lots, ub = 0.7, iterations = 10, model_maxiter = 100):
    simulation_var_x0_fx_iter = {}
    constraints, bnd = constraints_bounds(df_recipe, df_available_lots, ub = ub)
    for iteration in range(iterations):
        x0 = np.random.randint(0, df_available_lots['LOT_AVAILABILITY_KG'], len(df_available_lots['COST_BRL_KG']))
        simulation_var_x0_fx_iter[str(iteration)] = minimize_round(df_recipe, df_available_lots, constraints, bnd, x0 = x0, maxiter = model_maxiter), {'x0': x0} 
        logging.warning('Appended iteration %d', iteration)

    return simulation_var_x0_fx_iter

def model_results(df_recipe, df_available_lots, model_output):
	df_model_output = df_available_lots[['LOT_AVAILABILITY_KG']]
	df_model_output['MODEL_RESULT_ARRAY'] = model_output.x.astype(int)

	df_model_output['ALLOCATION_TEST'] = [True if df_model_output['MODEL_RESULT_ARRAY'][x] <= df_model_output['LOT_AVAILABILITY_KG'][x] else False for x in range(len(df_model_output['LOT_AVAILABILITY_KG']))]
	
	df_model_recipe_output = pd.DataFrame()
	df_model_recipe_output['AROMA'] = [abs(float(df_recipe['AROMA']) - (sum(np.array(df_available_lots['AROMA'])*model_output.x))/sum(model_output.x))]

	df_model_recipe_output['FLAVOR'] = [abs(float(df_recipe['FLAVOR']) - (sum(np.array(df_available_lots['FLAVOR'])*model_output.x))/sum(model_output.x))]

	df_model_recipe_output['AFTERTASTE'] = [abs(float(df_recipe['AFTERTASTE']) - (sum(np.array(df_available_lots['AFTERTASTE'])*model_output.x))/sum(model_output.x))]

	df_model_recipe_output['ACIDITY'] = [abs(float(df_recipe['ACIDITY']) - (sum(np.array(df_available_lots['ACIDITY'])*model_output.x))/sum(model_output.x))]

	df_model_recipe_output['BODY'] = [abs(float(df_recipe['BODY']) - (sum(np.array(df_available_lots['BODY'])*model_output.x))/sum(model_output.x))]

	df_model_recipe_output['BALANCE'] = [abs(float(df_recipe['BALANCE']) - (sum(np.array(df_available_lots['BALANCE'])*model_output.x))/sum(model_output.x))]

	df_model_recipe_output['MOISTURE'] = [abs(float(df_recipe['MOISTURE']) - (sum(np.array(df_available_lots['MOISTURE'])*model_output.x))/sum(model_output.x))]

	constraints_results = df_model_recipe_output
	
	deviation_percentage = 100*df_model_recipe_output/df_recipe[df_model_recipe_output.columns]

	return df_model_output, constraints_results, deviation_percentage