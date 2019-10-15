#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
from simCOSINOR.cynumstats import cy_lin_lstsqr_mat_residual,se_of_slope
from scipy.stats import t, f, norm

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

def periodogram():
	print("To-do")

def sliding_window_cosinor():
	print("To-do")

