#	MIT License
#
#	Copyright (c) 2017 Harsh Chaturvedi
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy
#	of this software and associated documentation files (the "Software"), to deal
#	in the Software without restriction, including without limitation the rights
#	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#	copies of the Software, and to permit persons to whom the Software is
#	furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all
#	copies or substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#	SOFTWARE.

'''
Minimization of least-squares distance between non-linear functions via 
optimization of free parameters (arguments).

In problems such as those of curve fitting and dynamical scaling, we often need 
to evaluate free parameters that will minimize the distance between different 
non-linear functions of some independent variable(s). The optimize_arguments 
function achieves this by performing an iterative grid-search over the space of 
free parameters (arguments) as specified by the user.
'''

import numpy as np
from math import fsum, fabs
from scipy.optimize import minimize

#sample output:
#{'args': {'off': -857.368421052539, 'po': -9.314108}, 'ys': [{'y':[1,2,3,4,...1000],'x_slope':1.0,'x_offset':0.0,'y_slope':1.0,'y_offset':0.0}, {'y':[-4,-5,-6,-7,...-560],'x_slope':1399.0898,'x_offset':128397.65347,'y_slope':-13.109283,'y_offset':1290.23}]}

def optimize_arguments(x_dicts, trial_args, args_deltas={}, ord_mag=1, delta_pieces=10, prec_iters=5):
	#x_dicts=[{'x':[1,2,3],'y_func':y_func1,'arg_inds':['a']}, {'x':[1,2,3]}, {'x':[2,3],'y_func':y_func2,'arg_inds':['a','c']}], trial_args={'a':50,'b':60,'c':2}, args_deltas={'a':5., 'b':11.}
	
	#self returning y_funcs for static xs
	def y_is_x(x):
		return np.array(x)
	for x_dict_ind in xrange(len(x_dicts)):
		if 'y_func' not in x_dicts[x_dict_ind]:
			x_dicts[x_dict_ind].update({'y_func':y_is_x})
		if 'arg_inds' not in x_dicts[x_dict_ind]:
			x_dicts[x_dict_ind].update({'arg_inds':[]})
			
	#making xs equal sized by truncation
	x_lens = []
	for x_dict in x_dicts:
		x_lens.append(len(x_dict['xs']))
	min_len = min(x_lens)
	for x_dict_ind in xrange(len(x_dicts)):
		x_dicts[x_dict_ind]['xs'] = x_dicts[x_dict_ind]['xs'][0:min_len]
		
	#updating scaled 'x' values
	def scale(x, home_points, away_points):
		slope = float(away_points['x1']-away_points['x2'])/float(home_points['x1']-home_points['x2'])
		offset = away_points['x1'] - slope*home_points['x1']
		x_scaled = slope*np.array(x) + offset
		return {'xs':x_scaled, 'slope':slope, 'offset':offset}
	for x_dict in x_dicts:
		x_dict.update({'x_slope':1.0, 'x_offset':0.0})
		if 'scaling_points' in x_dict:
			scale_dict = scale(x=x_dict['xs'], home_points={'x1':x_dict['scaling_points']['home_points']['x1'],'x2':x_dict['scaling_points']['home_points']['x2']}, away_points={'x1':x_dict['scaling_points']['away_points']['x1'],'x2':x_dict['scaling_points']['away_points']['x2']})
			x_dict['xs'], x_dict['x_slope'], x_dict['x_offset'] = scale_dict['xs'], scale_dict['slope'], scale_dict['offset']

	# keeping only finite values
	finite_ixs_list = [np.isfinite(x_dict['xs']) for \
		x_dict in x_dicts]
	finite_ixs = np.prod(finite_ixs_list, axis = 0).astype(bool)
	for x_dict in x_dicts:
		x_dict['xs'] = x_dict['xs'][finite_ixs]
		
	# initializing args properties
	arg_names = trial_args.keys()
	arg_guesses = [trial_args[name] for name in arg_names]
	arg_dels = [args_deltas[name] if name in args_deltas else \
					fabs(trial_args[name])*10.0**ord_mag for name in arg_names]
	arg_bounds = [(arg_guesses[ix] - arg_dels[ix], arg_guesses[ix] + \
					arg_dels[ix]) for ix in range(len(arg_guesses))]

	def get_ys_list(arg_arr, x_dicts = x_dicts, arg_names = arg_names):
		ys_list = []
		for x_dict in x_dicts:
			x_dict_arg_ixs = [arg_names.index(arg_ind) for \
				arg_ind in x_dict['arg_inds']]
			x_dict_args = arg_arr[x_dict_arg_ixs]
			ys = np.array(x_dict['y_func'](x_dict['xs'], *x_dict_args))
			ys_list.append(ys)
		return ys_list

	err_func = lambda arg_arr: np.sum(np.std(get_ys_list(arg_arr), axis = 0))

	#------------main----------------------

	best_args_arr = minimize(fun = err_func, x0 = arg_guesses, 
		bounds = arg_bounds, method = 'SLSQP', options = {'disp': True})['x']
	best_args_dict = dict([(arg_names[ix], best_args_arr[ix]) for ix in 
		range(len(arg_names))])
	best_ys_list = get_ys_list(best_args_arr)
	ret = {'args': best_args_dict, 'ys_list': best_ys_list, 
		'finite_ixs': finite_ixs}
	
	for y_ind in range(len(ret['ys_list'])): #include slopes and offsets in ret
		ys = ret['ys_list'][y_ind]
		x_dict = x_dicts[y_ind]
		ret['ys_list'][y_ind] = {'ys': ys, 
			'x_slope': x_dict['x_slope'],
			'x_offset': x_dict['x_offset'],
			'y_slope': 1.0,
			'y_offset': 0.0
			}
		if 'scaling_points' in x_dict:
			scale_dict = scale(x = ys, 
				home_points = {'x1': x_dict['scaling_points']['away_points']['y1'],
					'x2': x_dict['scaling_points']['away_points']['y2']
					}, 
				away_points = {'x1': x_dict['scaling_points']['home_points']['y1'],
					'x2': x_dict['scaling_points']['home_points']['y2']
					}
				)
			ret['ys_list'][y_ind].update({'ys': scale_dict['xs'],
				'y_slope': scale_dict['slope'], 
				'y_offset': scale_dict['offset']
				})

	return ret

#SAMPLE USE [for instruction purposes only]
'''
def y_func(x, po, off):	
	return po**np.array(x) + off
	
print optimize_arguments(x_dicts=[{'x':-9.314108**np.arange(1,30)}, {'x':np.arange(1,30),'y_func':y_func,'arg_inds':['po', 'off']}, 'scaling_points':{'home_points':{'x1':1.0,'y1':2.0,'x2':3.0,'y2':4.0},'away_points':{'x1':1.0,'y1':2.0,'x2':3.0,'y2':4.0}}], trial_args={'po':1,'off':90})

#output:
#{'args': {'off': -857.368421052539, 'po': -9.314108}, 'ys_list': [{'ys':[1,2,3,4,...1000],'x_slope':1.0,'x_offset':0.0,'y_slope':1.0,'y_offset':0.0}, {'ys':[-4,-5,-6,-7,...-560],'x_slope':1399.0898,'x_offset':128397.65347,'y_slope':-13.109283,'y_offset':1290.23}]}
'''
