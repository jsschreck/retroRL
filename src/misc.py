import cPickle as pickle 
from bson.binary import Binary
from scipy.sparse import csr_matrix, vstack as sparse_vstack 
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
from functools import partial
from retroRL.src.cost import Reset, MinCost

import numpy as np, glob 
import gzip, time, random, shutil, itertools 
import os, traceback, sys 

def mol_to_fp(mol, FINGERPRINT_SIZE = 16384, FP_rad = 3):
    if mol is None:
        return np.zeros((FINGERPRINT_SIZE,), dtype=np.int)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, FP_rad, nBits=FINGERPRINT_SIZE,
                                        useChirality=True, useFeatures=False), dtype=np.int)

def get_feature_vec(smiles, FINGERPRINT_SIZE, FP_rad):
    if not smiles:
        return np.zeros((FINGERPRINT_SIZE,), dtype=np.int)
    return mol_to_fp(Chem.MolFromSmiles(smiles), FINGERPRINT_SIZE, FP_rad)

def arr_for_cost(cost, SCALE = 1.0):
	val = np.array((abs(float(cost)),))
	return val

def get_states(pair, FINGERPRINT_SIZE, FP_rad, max_depth = 10, SCALE = 1.0):
	_id, smiles, cost, is_buyable = pair
	_id = int(_id)
	fps = np.hstack([get_feature_vec(smiles, FINGERPRINT_SIZE, FP_rad), max_depth-d])
	carr = arr_for_cost(cost, SCALE)
	return (smiles, _id, fps, carr, is_buyable)

def save_sparse_tree(smile, smile_id, depth, game, array, value, fid, FP_size):   
	value, min_value, converged = value
	array = csr_matrix(array) 
	matrix_parameters = [array.data, array.indices, array.indptr, array.shape, smile, smile_id, depth, game, value, min_value, converged]
	pickle.dump(matrix_parameters,fid,pickle.HIGHEST_PROTOCOL)

def save_mongo_doc(smile, depth, fps, cost, fid):
	fps = Binary(pickle.dumps(fps, protocol=pickle.HIGHEST_PROTOCOL))
	record = {'_id': str(smile), 'depth': int(depth), 'cost': float(cost), 'fps_16384': fps}
	pickle.dump(record,fid,pickle.HIGHEST_PROTOCOL)

def value_network_training_states(smiles_id, training_states, starting_chemicals, max_depth = 10, FP_rad = 3, FPS_size = 16384, fileName = "", mongoName = ""):
	saved = 0 
	buyables = 0 
	didworse = 0  
	resampled = 0
	new_states = 0
	finite_cost = 0 
	states_with_new_values = 0 
	
	#no_buyables = len([chem_cost for (chem_key,chem_cost) in training_states.items() if chem_cost == 0.0]) 
	if mongoName:
		gid = open(mongoName, "a+b")
	with open(fileName, "a+b") as fid:
		already_added = []
		for chem_key, chem_cost in training_states.items():
			already_seen = False
			if chem_key in starting_chemicals:
				if chem_cost < starting_chemicals[chem_key]:
					states_with_new_values += 1
				elif chem_cost == starting_chemicals[chem_key]:
					if random.random() < 0.9: # Resample with p = 0.1 ?
						continue
					resampled += 1
					already_seen = True
				else: # Sampled cost higher than what we've seen -> retrain the state.
					resampled += 1
					didworse += 1 
					chem_cost = starting_chemicals[chem_key]
					already_seen = True
			else:
				new_states += 1
			###
			if chem_cost == 0.0:# and : 
				buyables += 1 
			###
			smi, depth = chem_key
			fps = np.hstack([get_feature_vec(smi, FPS_size, FP_rad), max_depth - int(depth)])
			cost_arr = arr_for_cost(float(chem_cost))
			if not already_seen and mongoName:
				save_mongo_doc(smi,depth,fps,chem_cost,gid)
			already_seen = False
			save_sparse_tree(smi,smiles_id,depth,fps,cost_arr,fid,FPS_size)
			saved += 1
			if chem_cost < 20:
				finite_cost += 1
	if mongoName:
		gid.close()
	#print "... did worse on {} chemicals" .format(didworse)
	print "... saved {} buyables states".format(buyables)
	#print "... saved {} states with new values".format(states_with_new_values)
	#print "... saved {} entirely new states".format(new_states)
	print "... saved {} states with finite cost" .format(finite_cost)
	print "... saved {} TOTAL states for training.".format(saved)

def greedy_training_states(crns, smiles_id, max_depth = 10, game_no = 0, FP_rad = 3, FPS_size = 16384, fileName = "", tolerance = 0.1, verbose = False):
	saved = 0 
	buyables = 0 
	failures = 0
	resampled = 0 
	successes = 0 
	converged = 0 
	penalty_cost = 0 	
	C_1, R_1, C_0, R_0 = crns

	if not len(C_0): 
		ave_cost_0 = 0.0

	for chem_key in C_1:
		smi, depth = chem_key 
		smi_converged = False

		if not len(C_1[chem_key].rewards): 
			continue

		'''
		X1 = [abs(x) for x in C_1[chem_key].rewards]
		mincost_1 = min(X1)
		ave_cost_1 = np.mean(X1)
		ave_cost_std_1 = np.std(X1)
		N1 = len(X1)
		'''

		N0 = 0
		ave_cost_0 = float("inf")  		
		if chem_key in C_0:
			if len(C_0[chem_key].rewards):
				X0 = [abs(x) for x in C_0[chem_key].rewards]
				ave_cost_0 = np.mean(X0)
				N0 = len(X0)

		X1 = [abs(x) for x in C_1[chem_key].rewards][N0:]
		mincost_1 = min(C_1[chem_key].rewards)
		ave_cost_1 = np.mean(X1)
		ave_cost_std_1 = np.std(X1)
		N1 = len(X1)

		if depth == 0:
			target_key = chem_key
			target_cost_0 = ave_cost_0
			target_cost_1 = ave_cost_1
			target_cost_std_1 = ave_cost_std_1

		if ave_cost_1 < 1e-3:
			buyables += 1 
		if ave_cost_1 < 20:
			successes += 1 
		else:
			failures += 1

		if int(depth) == 10 and ave_cost_1 == 100.0: continue

		if abs(ave_cost_1 - ave_cost_0) < tolerance:
			converged += 1
			smi_converged = True 

		visits = N1 - N0
		fps = np.hstack([get_feature_vec(smi, FPS_size, FP_rad), max_depth - int(depth)])
		save_sparse_tree(smi,smiles_id,depth,visits,fps,[ave_cost_1,mincost_1,smi_converged],fileName,FPS_size)
		saved += 1
	
	if verbose:
		print "----- {} -----".format(smiles_id)
		print "... saved {} buyables states".format(buyables)
		print "... saved {} states with flat average cost".format(resampled)
		print "... saved {} states with ave-cost < 20" .format(successes)
		print "... saved {} TOTAL states for training.".format(failures+successes)
		print "... |C0| = {} |C1| = {}".format(len(C_0),len(C_1))
		if np.isfinite(target_cost_0):
			diff = target_cost_1 - target_cost_0
		else:
			diff = target_cost_1
		print "... Target cost: {} -> {} +/- {}, delta = {}".format(target_cost_0,target_cost_1,target_cost_std_1,diff)
	
def pathway_training_states(crns, smiles_id, max_depth = 10, FP_rad = 3, FPS_size = 16384, fileName = "", tolerance = 0.01, verbose = False):
	pass 

if __name__ == '__main__':
	replica = int(sys.argv[1])
	update_values(replica)
