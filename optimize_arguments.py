import numpy as np
from math import fsum, fabs

#output:
#{'args': {'off': -857.368421052539, 'po': -9.314108}, 'ys': [{'y':[1,2,3,4,...1000],'x_slope':1.0,'x_offset':0.0,'y_slope':1.0,'y_offset':0.0}, {'y':[-4,-5,-6,-7,...-560],'x_slope':1399.0898,'x_offset':128397.65347,'y_slope':-13.109283,'y_offset':1290.23}]}

def optimize_arguments(x_dicts, trial_args, args_deltas={}, ord_mag=1, delta_pieces=10, prec_iters=10):
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
		x_lens.append(len(x_dict['x']))
	min_len = min(x_lens)
	for x_dict_ind in xrange(len(x_dicts)):
		x_dicts[x_dict_ind]['x'] = x_dicts[x_dict_ind]['x'][0:min_len]
		
	#updating scaled 'x' values
	def scale(x, home_points, away_points):
		slope = float(away_points['x1']-away_points['x2'])/float(home_points['x1']-home_points['x2'])
		offset = away_points['x1'] - slope*home_points['x1']
		x_scaled = slope*np.array(x) + offset
		return {'x':x_scaled, 'slope':slope, 'offset':offset}
	for x_dict in x_dicts:
		x_dict.update({'x_slope':1.0, 'x_offset':0.0})
		if 'scaling_points' in x_dict:
			scale_dict = scale(x=x_dict['x'], home_points={'x1':x_dict['scaling_points']['home_points']['x1'],'x2':x_dict['scaling_points']['home_points']['x2']}, away_points={'x1':x_dict['scaling_points']['away_points']['x1'],'x2':x_dict['scaling_points']['away_points']['x2']})
			x_dict['x'], x_dict['x_slope'], x_dict['x_offset'] = scale_dict['x'], scale_dict['slope'], scale_dict['offset']
		
	#initializing args_dicts_list eg args_dict_list = [{'name':'a', 'guess':1.0, 'delta':100}, {'name':'b', 'guess':1.0,'delta':100.0}, {'name':'c', 'guess':15.0,'delta':0.0035}]
	args_dict_list = [{'name':key, 'guess':trial_args[key], 'delta': (key in args_deltas and args_deltas[key]) or fabs(trial_args[key])*10.0**ord_mag} for key in trial_args]
	
	def get_args_set():
		args_set = [] #[[53.0,0.0074], [2111.0,6.0]]
		nu_args = len(args_dict_list)
		loop_info = {'nu_loops':nu_args, 'nu_iters':2*delta_pieces+1, 'iter_nus':[0]*nu_args}
		while True:
			#do the task
			args = {}
			for dict_ind in xrange(nu_args):
				args_dict = args_dict_list[dict_ind]
				name, guess, delta, delta_piece_nu = args_dict['name'], args_dict['guess'], args_dict['delta'], loop_info['iter_nus'][dict_ind]
				arg = guess-delta+delta/delta_pieces*delta_piece_nu
				args.update({name:arg})
			args_set.append(args)
			#increment the counter
			done_flag = False
			loop_ind = 0
			while True:
				if loop_info['iter_nus'][loop_ind] < loop_info['nu_iters']-1: #if iter_nu<last_iter
					loop_info['iter_nus'][loop_ind] += 1
					for lower_ind in xrange(loop_ind): #make all lower loop iters 0
						loop_info['iter_nus'][lower_ind] = 0
					break
				elif loop_ind < loop_info['nu_loops']-1: #go to higher loop
					loop_ind += 1
				else:
					done_flag = True
					break
			if done_flag == True:
				break
		return args_set
		
	def get_best_args(args_set):
		def get_ys(args):
			ys = []
			for x_dict in x_dicts:
				x_dep_args = [args[arg_ind] for arg_ind in x_dict['arg_inds']]
				y = np.array(x_dict['y_func'](x_dict['x'], *x_dep_args))
				ys.append(y)
			return ys
		total_distance = lambda ys: np.sum(np.array(ys).std(0))
		best_args, best_distance = args_set[0], total_distance(get_ys(args_set[0]))
		for args_ind in np.arange(1,len(args_set)):
			distance = total_distance(get_ys(args_set[args_ind]))
			if distance < best_distance:
				best_args, best_distance = args_set[args_ind], distance
		best_ys = get_ys(best_args)
		return {'args':best_args, 'ys':best_ys}
		
	def update_args_dict_list(best_args):
		for dict_ind in xrange(len(args_dict_list)):
			args_dict_list[dict_ind]['guess'] = best_args[args_dict_list[dict_ind]['name']]
			args_dict_list[dict_ind]['delta'] = args_dict_list[dict_ind]['delta']/delta_pieces
			
	def main():
		best_args = {}
		for prec_ind in xrange(prec_iters):
			best_args = get_best_args(get_args_set()) #get a set of possible arguments based on previous guesses
			update_args_dict_list(best_args['args'])
		for y_ind in range(len(best_args['ys'])): #include slopes and offsets in best_args
			best_args['ys'][y_ind] = {'y':best_args['ys'][y_ind],'x_slope':x_dicts[y_ind]['x_slope'],'x_offset':x_dicts[y_ind]['x_offset'],'y_slope':1.0,'y_offset':0.0}
			x_dict = x_dicts[y_ind]
			if 'scaling_points' in x_dict:
				scale_dict = scale(x=best_args['ys'][y_ind]['y'], home_points={'x1':x_dict['scaling_points']['away_points']['y1'],'x2':x_dict['scaling_points']['away_points']['y2']}, away_points={'x1':x_dict['scaling_points']['home_points']['y1'],'x2':x_dict['scaling_points']['home_points']['y2']})
				best_args['ys'][y_ind].update({'y':scale_dict['x'], 'y_slope':scale_dict['slope'], 'y_offset':scale_dict['offset']})
		return best_args
		
	return main() #execute main()

#SAMPLE USE [for instruction purpose only]
'''
def y_func(x, po, off):	
	return po**np.array(x) + off
	
print optimize_arguments(x_dicts=[{'x':-9.314108**np.arange(1,30)}, {'x':np.arange(1,30),'y_func':y_func,'arg_inds':['po', 'off']}, 'scaling_points':{'home_points':{'x1':1.0,'y1':2.0,'x2':3.0,'y2':4.0},'away_points':{'x1':1.0,'y1':2.0,'x2':3.0,'y2':4.0}}], trial_args={'po':1,'off':90})

#output:
#{'args': {'off': -857.368421052539, 'po': -9.314108}, 'ys': [{'y':[1,2,3,4,...1000],'x_slope':1.0,'x_offset':0.0,'y_slope':1.0,'y_offset':0.0}, {'y':[-4,-5,-6,-7,...-560],'x_slope':1399.0898,'x_offset':128397.65347,'y_slope':-13.109283,'y_offset':1290.23}]}
'''
