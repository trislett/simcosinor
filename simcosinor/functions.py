#!/usr/bin/env python

import os
import numpy as np
from simcosinor.cynumstats import cy_lin_lstsqr_mat_residual, cy_lin_lstsqr_mat, se_of_slope
from scipy.stats import t, f, norm

def example_file():
	return("%s/simcosinor/examples/examples_subjects_norm.csv" % os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def run_cosinor_simulation(endog, time_variable, period = [24.0], resids = None, randomise_time = False, resample_eveningly = False, n_sampling = None, range_sampling = None, i = 0):
	n = len(endog)
	k = len(period)*2 + 1
	DF_Between = k - 1 # aka df model
	DF_Within = n - k # aka df residuals

	# Check that endog has two dimensions
	if endog.ndim == 1:
		endog = endog.reshape(len(endog),1)

	# calculate residuals from cosinor model if not already provided
	if resids is None:
		resids = residual_cosinor(endog = endog, time_var = time_variable, period = period)

	# Calculate true Mesor, Amplitude, Acrophase
	MESOR, AMPLITUDE, ACROPHASE = glm_cosinor(endog = endog, 
															time_var = time_variable,
															period = [24.0],
															calc_MESOR = True,
															output_fit_only = True)

	if randomise_time:
		if n_sampling is None:
			n_sampling = n
		if range_sampling is None:
			range_sampling = [0,23.99]
		if resample_eveningly:
			time_variable = np.linspace(range_sampling[0],range_sampling[1],n_sampling)
		else:
			time_variable = np.sort(np.random.uniform(low=range_sampling[0], high=range_sampling[1], size=(n_sampling,)))

	# the mean and std of for the noise is calculated from the residuals
	noise_mean = resids.mean()
	noise_std = resids.std()
	noise_npts = n_sampling
	noise = np.random.normal(noise_mean, noise_std, noise_npts).reshape(noise_npts,1)

	# calculate the predicted cosinor curve
	predicted = project_cosionor_model(MESOR, AMPLITUDE, ACROPHASE, TIME_VAR = time_variable, PERIOD = period)
	sim_endog = noise + predicted
	sim_R2, sim_MESOR, _, sim_AMPLITUDE, sim_SE_AMPLITUDE, sim_ACROPHASE, sim_SE_ACROPHASE, sim_Fmodel, _, sim_tAMPLITUDE, _, _ = glm_cosinor(endog = sim_endog, 
																time_var = time_variable,
																period = period,
																calc_MESOR = True,
																output_fit_only = False)

	ACROPHASE_24 = np.zeros_like(sim_ACROPHASE)
	for j, per in enumerate(period):
		acrotemp = np.abs(sim_ACROPHASE[j]/(2*np.pi)) * per
		acrotemp[acrotemp>per] -= per
		ACROPHASE_24[j] = acrotemp

	p_values = f.sf(sim_Fmodel, DF_Between, DF_Within)
	return(sim_R2.squeeze(), sim_Fmodel.squeeze(), sim_tAMPLITUDE.squeeze(), ACROPHASE_24.squeeze(), p_values.squeeze())

def regression_f_ratio(endog, exog_m1, exog_m2, covars = None):
	"""
	Compares regression models
	
	Parameters
	----------
	endog : array
		Endogenous (dependent) variable array (Nsubjects, Nvariables)
	exog_m1 : array
		Exogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects)
	exog_m2 : array
		Exogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects)
	covars : array
		Dummy coded array of covariates of no interest
	
	Returns
	---------
	F_ratio : array
		F distribution with dof (dfN, dfF)
	"""

	n = endog.shape[0]
	# model1
	k_m1 = exog_m1.shape[1]
	DF_Between_m1 = k_m1 - 1 # aka df model
	DF_Within_m1 = n - k_m1 # aka df residuals
	_, SS_Residuals_m1 = cy_lin_lstsqr_mat_residual(exog_m1,endog)

	k_m2 = exog_m2.shape[1]
	DF_Between_m2 = k_m2 - 1 # aka df model
	DF_Within_m2 = n - k_m2 # aka df residuals
	_, SS_Residuals_m2 = cy_lin_lstsqr_mat_residual(exog_m2,endog)

	# Just to make things easier
	dfN = DF_Within_m2 - DF_Within_m1
	dfF = DF_Within_m1
	dfR = DF_Within_m2

	F_ratio = (((SS_Residuals_m2 - SS_Residuals_m1)/(dfR - dfF))/(SS_Residuals_m1/dfF))
	p_values = f.sf(F_ratio, dfN, dfF)
	return(F_ratio, p_values)


# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3991883/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3663600/
def glm_cosinor(endog, time_var, exog = None, dmy_covariates = None, rand_array = None, interaction_var = None, period = [24.0], calc_MESOR = True, output_fit_only = False):
	"""
	COSINOR model using GLM
	
	Parameters
	----------
	endog : array
		Endogenous (dependent) variable array (Nsubjects, Nvariables)
	time_var : array
		Time variable [0-23.99] (Nsubjects).
	exog : array
		Exogenous (independent) dummy coded variables
		exog is an array of arrays (Nvariables, Nsubjects, Kvariable).
	dmy_covariates : array
		Dummy coded array of covariates of no interest.
	init_covars : array
		Dummy coded array of covariates for two-step regression.
	rand_array : array
		randomized array for permutations (Nsubjects).
	period : array
		Period(s) as an array of floats for cosinor model.
	Returns
	---------
	To-do
	"""

	n = endog.shape[0]
	# add cosinor terms
	num_period = len(period)
	exog_vars = np.ones((n))
	for i in range(num_period):
		exog_vars = np.column_stack((exog_vars,np.cos(np.divide(2.0*np.pi*time_var, period[i]))))
		exog_vars = np.column_stack((exog_vars,np.sin(np.divide(2.0*np.pi*time_var, period[i]))))

	if interaction_var is not None:
		for i in range(num_period):
			exog_vars = np.column_stack((exog_vars, exog_vars[i+1] * interaction_var))


	kvars = []
	# add other exogenous variables to the model (currently not implemented)
	if exog is not None:
		for var in exog:
			var = np.array(var)
			if var.ndim == 1:
				kvars.append((3))
			else:
				kvars.append((var.shape[1]))
			exog_vars = np.column_stack((exog_vars,var))

	# add covariates (i.e., exogenous variables that will not be outputed)
	if dmy_covariates is not None:
		exog_vars = np.column_stack((exog_vars, dmy_covariates))
	exog_vars = np.array(exog_vars)

	if rand_array is not None:
		exog_vars = exog_vars[rand_array]

	# calculate model fit (Fmodel and R-sqr)
	k = exog_vars.shape[1]
	DF_Between = k - 1 # aka df model
	DF_Within = n - k # aka df residuals
	#DF_Total = n - 1

	a, SS_Residuals = cy_lin_lstsqr_mat_residual(exog_vars,endog)
	if output_fit_only:
		AMPLITUDE = []
		ACROPHASE = []
		MESOR = a[0]
		for i in range(num_period):
			# beta, gamma
			AMPLITUDE.append(np.sqrt((a[1+(i*2),:]**2) + (a[2+(i*2),:]**2)))
			# Acrophase calculation
			if i == 0: # awful hack
				ACROPHASE = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				ACROPHASE = ACROPHASE[np.newaxis,:]
			else:
				temp_acro = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				temp_acro = temp_acro[np.newaxis,:]
				ACROPHASE = np.append(ACROPHASE,temp_acro, axis=0)
			ACROPHASE = np.array(ACROPHASE)
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)] = -ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)]
			ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)] = (-1*np.pi) + ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)]
			ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)] = (-1*np.pi) - ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)]
			ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)] = (-2*np.pi) + ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)]
		return MESOR, np.array(AMPLITUDE), np.array(ACROPHASE)
	else:
		SS_Total = np.sum((endog - np.mean(endog,0))**2,0)
		SS_Between = SS_Total - SS_Residuals
		MS_Residuals = (SS_Residuals / DF_Within)
		Fmodel = (SS_Between/DF_Between) / MS_Residuals
		# Calculates sigma sqr and T-value (intercept) for MESOR
		sigma = np.sqrt(SS_Residuals / DF_Within)
		invXX = np.linalg.inv(np.dot(exog_vars.T, exog_vars))

		if (calc_MESOR) or (exog is not None):
			if endog.ndim == 1:
				se = np.sqrt(np.diag(sigma * sigma * invXX))
				Tvalues = a / se
				MESOR = a[0]
				tMESOR = Tvalues[0]
				SE_MESOR = se[0]
				a = a[:, np.newaxis]
			else:
				num_depv = endog.shape[1]
				se = se_of_slope(num_depv,invXX,sigma**2,k)
				Tvalues = a / se
				MESOR = a[0,:]
				tMESOR = Tvalues[0,:]
				SE_MESOR = se[0,:]
			if exog is not None:
				tEXOG = Tvalues[(3+(2*(num_period-1))):,:]
			else:
				tEXOG = None
		else:
			MESOR = tMESOR = SE_MESOR = tEXOG = None

		AMPLITUDE = []
		ACROPHASE = []
		SE_ACROPHASE = []
		SE_AMPLITUDE = []
		tAMPLITUDE = []
		tACROPHASE = []
		
		for i in range(num_period):
			# beta, gamma
			AMPLITUDE.append(np.sqrt((a[1+(i*2),:]**2) + (a[2+(i*2),:]**2)))
			# Acrophase calculation
			if i == 0: # awful hack
				ACROPHASE = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				ACROPHASE = ACROPHASE[np.newaxis,:]
			else:
				temp_acro = np.arctan(np.abs(np.divide(-a[2+(i*2),:], a[1+(i*2),:])))
				temp_acro = temp_acro[np.newaxis,:]
				ACROPHASE = np.append(ACROPHASE,temp_acro, axis=0)

			# standard errors from error propagation
			SE_ACROPHASE.append(sigma * np.sqrt((invXX[(1+(i*2)),1+(i*2)]*np.sin(ACROPHASE[i])**2) + (2*invXX[1+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])*np.cos(ACROPHASE[i])) + (invXX[2+(i*2),2+(i*2)]*np.cos(ACROPHASE[i])**2)) / AMPLITUDE[i])
			SE_AMPLITUDE.append(sigma * np.sqrt((invXX[(1+(i*2)),1+(i*2)]*np.cos(ACROPHASE[i])**2) - (2*invXX[1+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])*np.cos(ACROPHASE[i])) + (invXX[2+(i*2),2+(i*2)]*np.sin(ACROPHASE[i])**2)))

			ACROPHASE = np.array(ACROPHASE)
			if rand_array is None:
				ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)] = -ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] >= 0)]
				ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)] = (-1*np.pi) + ACROPHASE[i, (a[2+(i*2),:] > 0) & (a[1+(i*2),:] < 0)]
				ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)] = (-1*np.pi) - ACROPHASE[i, (a[2+(i*2),:] < 0) & (a[1+(i*2),:] <= 0)]
				ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)] = (-2*np.pi) + ACROPHASE[i, (a[2+(i*2),:] <= 0) & (a[1+(i*2),:] > 0)]
			# t values
			tAMPLITUDE.append(np.divide(AMPLITUDE[i], SE_AMPLITUDE[i]))
			tACROPHASE.append(np.divide(1.0, SE_ACROPHASE[i]))

		# Do not output R-squared during permutations testing.
		R2 = 1 - (SS_Residuals/SS_Total)

		return R2, MESOR, SE_MESOR, np.array(AMPLITUDE), np.array(SE_AMPLITUDE), np.array(ACROPHASE), np.array(SE_ACROPHASE), Fmodel, tMESOR, np.abs(tAMPLITUDE), np.abs(tACROPHASE), np.array(tEXOG)

def lm_residuals(endog, exog):
	"""
	"""
	if exog.ndim == 1:
		exog = stack_ones(exog)
	if np.mean(exog[:,0]) != 1:
		exog = stack_ones(exog)
	a = cy_lin_lstsqr_mat(exog,endog)
	endog = endog - np.dot(exog,a)
	return endog

def project_cosionor_model(MESOR, AMPLITUDE, ACROPHASE, TIME_VAR, PERIOD = [24.0]):
	TIME_VAR = np.array(TIME_VAR)
	n = len(TIME_VAR)
	r = len(MESOR)
	proj = MESOR
	for j, per in enumerate(PERIOD):
		proj = proj + AMPLITUDE[j,:]*np.cos((np.divide(2*np.pi*np.tile(TIME_VAR,r).reshape(r,n).T, per) + ACROPHASE[j,:]))
	return proj

def residual_cosinor(endog, time_var, period = [24.0]):
	n = endog.shape[0]
	num_period = len(period)
	exog_vars = np.ones((n))
	for i in range(num_period):
		exog_vars = np.column_stack((exog_vars,np.cos(np.divide(2.0*np.pi*time_var, period[i]))))
		exog_vars = np.column_stack((exog_vars,np.sin(np.divide(2.0*np.pi*time_var, period[i]))))
	return np.array(lm_residuals(endog, exog_vars))




def periodogram():
	print("To-do")

def sliding_window_cosinor():
	print("To-do")

def check_columns(pdData):
	for counter, roi in enumerate(pdData.columns):
		if counter == 0:
			num_subjects = len(pdData[roi])
		a = np.unique(pdData[roi])
		num_missing = np.sum(pdData[roi].isnull()*1)
		if len(a) > 10:
			astr = '[n>10]'
		else:
			astr = ','.join(a.astype(np.str))
			astr = '['+astr+']'
		if num_missing == 0:
			print("[%d] : %s\t%s" % (counter, roi, astr))
		else:
			print("[%d] : %s\t%s\t\tCONTAINS %d MISSING VARIABLES!" % (counter, roi, astr, num_missing))

