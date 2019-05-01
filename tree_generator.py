from makeit.retrosynthetic.transformer import RetroTransformer
from makeit.retrosynthetic.results import RetroResult, RetroPrecursor
from makeit.utilities.buyable.pricer import Pricer
from multiprocessing import Process, Manager, Queue
from multiprocessing import Pool
from pymongo import MongoClient
import gc as gc_gzip

from makeit.mcts.cost import Reset, score_max_depth, score_no_templates, MinCost, BuyablePathwayCount
from makeit.mcts.misc import get_feature_vec, save_sparse_tree
from makeit.mcts.misc import greedy_training_states, value_network_training_states
from makeit.mcts.nodes import Chemical, Reaction
from makeit.mcts.smiles_loader import smilesLoader

import makeit.global_config as gc
from functools import partial
import Queue as VanillaQueue
import multiprocessing as mp
import cPickle as pickle
import numpy as np
import traceback
import itertools
import datetime
import argparse
import psutil
import shutil
import random
import time
import gzip
import copy
import sys
import os
import io

NCPUS = psutil.cpu_count()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['openmp'] = 'True'
os.environ['MKL_NUM_THREADS'] = '{}'.format(NCPUS)
os.environ['GOTO_NUM_THREADS'] = '{}'.format(NCPUS)
os.environ['OMP_NUM_THREADS'] = '{}'.format(NCPUS)

class TreeGenerator:

	def __init__(self, load_transformer=True, pricer=None,
				 max_branching=50, total_applied_templates = 1000, max_depth=10,
				 nproc=8, mincount=25, chiral=True, template_prioritization=gc.relevance,
				 precursor_prioritization=gc.heuristic, max_ppg=100,
				 mincount_chiral=10, verbose = False,
				 filePath = "./",
				 mols_fileName = "output/crn/",
				 logger_fileName = "output/logs/logger.txt",
				 pathway_fileName = "output/pathways/pathways.pkl",
				 outcomes_fileName  = "output/logs/outcomes.txt",
				 training_states_fileName = "output/states/states.pickle"):

		self.nproc = nproc
		self.verbose = verbose
		self.mincount = mincount
		self.mincount_chiral = mincount_chiral

		self.max_depth = max_depth
		self.max_branching = max_branching
		self.total_applied_templates = total_applied_templates

		self.template_prioritization = template_prioritization
		self.precursor_prioritization = precursor_prioritization
		self.chiral = chiral

		self.filePath = filePath
		self.mols_fileName = mols_fileName
		self.logger_fileName = logger_fileName
		self.pathway_fileName = pathway_fileName
		self.outcomes_fileName  = outcomes_fileName
		self.training_states_fileName = training_states_fileName

		## Pricer
		if pricer:
			self.pricer = pricer
		else:
			self.pricer = Pricer()
		self.pricer.load_from_file()

		self.reset()

		self.retroTransformer = None

		## Load template transformer using Make-It.
		db_client = MongoClient(gc.MONGO['path'], gc.MONGO['id'],
								connect=gc.MONGO['connect']
								)
		TEMPLATE_DB = db_client[gc.RETRO_TRANSFORMS_CHIRAL['database']][gc.RETRO_TRANSFORMS_CHIRAL['collection']]
		self.retroTransformer = RetroTransformer(
			mincount=self.mincount, mincount_chiral=self.mincount_chiral,
			TEMPLATE_DB=TEMPLATE_DB
			)
		self.retroTransformer.chiral = self.chiral
		home = os.path.expanduser('~')
		if home.split("/")[1] == "rigel":
			home = "/rigel/cheme/users/jss2278/chemical_networks"
		transformer_filepath = home + "/Make-It/makeit/data/"
		if os.path.isfile(transformer_filepath+"chiral_templates.pickle"):
			self.retroTransformer.load_from_file(
				True, "chiral_templates.pickle", chiral=self.chiral, rxns=True,
				file_path=transformer_filepath
				)
		else:
			self.retroTransformer.dump_to_file(
				True, "chiral_templates.pickle", chiral=self.chiral,
				file_path=transformer_filepath
				)

		# Define method to start up parallelization.
		def prepare():
			print 'Tree builder spinning off {} child processes'.format(self.nproc)
			#MyLogger.print_and_log('Tree builder spinning off {} child processes'.format(
			#	self.nproc), treebuilder_loc)
			for i in range(self.nproc):
				p = Process(target=self.work, args=(i,))
				self.workers.append(p)
				p.start()
		self.prepare = prepare

		# Define method to check if all results processed
		def waiting_for_results():
			waiting = [expansion_queue.empty()
			           for expansion_queue in self.expansion_queues]
			for results_queue in self.results_queues:
				waiting.append(results_queue.empty())
			waiting += self.idle
			return (not all(waiting))
		self.waiting_for_results = waiting_for_results

		# Define method to get a processed result.
		def get_ready_result():
			for results_queue in self.results_queues:
				while not results_queue.empty():
					yield results_queue.get(timeout=0.1)
		self.get_ready_result = get_ready_result

		# Define method to get a signal to start a new attempt.
		def get_pathway_result():
			while not self.pathways_queue.empty():
				yield self.pathways_queue.get(timeout=0.2)
		self.get_pathway_result = get_pathway_result

		# Define how first target is set.
		def set_initial_target(_id,leaves):
			for leaf in leaves:
				self.active_chemicals[_id].append(leaf)
				self.expand_products(_id, [leaf], self.expansion_branching)
		self.set_initial_target = set_initial_target

		# Define method to add targets to the expansion queue
		def expand(_id, chem_smi, depth, branching):
			precursors = False
			if self.crns[_id][0][chem_smi,depth].retro_results != None:
				precursors = self.crns[_id][0][chem_smi,depth].retro_results
			self.expansion_queues[_id].put((_id, chem_smi, depth, branching, precursors))
		self.expand = expand

		# Define method to stop working.
		def stop():
			if not self.running:
				return
			self.done.value = 1
			#MyLogger.print_and_log('Terminating tree building process.', treebuilder_loc)
			for p in self.workers:
				if p and p.is_alive():
					p.terminate()
			#MyLogger.print_and_log('All tree building processes done.', treebuilder_loc)
			self.running = False
		self.stop = stop

	# Has no self dependencies, move to another script in future
	def ResetCostEstimate(self, Chemicals, Reactions, reset_only_estimate = False):
		for c in Chemicals.values():
			if len(c.rewards):
				c.cost_estimate = np.mean(c.rewards)
			else:
				try:
					if np.isfinite(c.cost_estimate):
						pass
					else:
						c.cost_estimate = float("inf")
				except:
					c.cost_estimate = float("inf")

		for r in Reactions.values():
			if len(r.rewards):
				r.cost_estimate = np.mean(r.rewards)
			else:
				try:
					if np.isfinite(r.cost_estimate):
						pass
					else:
						r.cost_estimate = float("inf")
				except:
					r.cost_estimate = float("inf")

	# pricer for self
	def get_price(self, chem_smi):
		ppg = self.pricer.lookup_smiles(chem_smi, alreadyCanonical=True)
		if ppg:
			return 0.0
		else:
			return None

	# model for self
	def estimate_current_cost(self):
		from makeit.prioritization.precursors.mincost import MinCostPrecursorPrioritizer
		model = MinCostPrecursorPrioritizer()
		model.load_model(datapath = self.model_weights + '/')
		return model

	# Has not many self dependencies, maybe move to another script in future
	def load_crn(self, targets):
		crns = []
		counter = 1
		for lst in targets:
			try:
				smiles_id, smiles = lst
			except:
				group_ID, smiles_id, smiles = lst
			mol_file = self.mols_fileName + "mol_{}.pkl".format(smiles_id)
			loaded = False

			if os.path.isfile(mol_file) and os.path.getsize(mol_file):
				try:
					gc_gzip.disable()
					with gzip.open(mol_file, "rb") as fod:
						smiles_id, smiles, C, R = pickle.load(
													io.BufferedReader(fod)
													)
						loaded = True
						gc_gzip.enable()
						if len(C[smiles,0].rewards) >= self.total_games:
							continue
						crns.append([counter, [smiles_id, smiles, C, R]])
						counter += 1
				except Exception as E:
					pass

				if not loaded:
					try:
						with open(mol_file, "rb") as fod:
							smiles_id, smiles, C, R = pickle.load(fod)
							if len(C[smiles,0].rewards) >= self.total_games:
								continue
							crns.append([counter, [smiles_id, smiles, C, R]])
							counter += 1
					except Exception as E:
						crns.append([counter, [smiles_id, smiles, {}, {}]])
						counter += 1
			else:
				crns.append([counter, [smiles_id, smiles, {}, {}]])
				counter += 1

			if len(crns) == self.group_size:
				yield crns
				crns = []
				counter = 1

		yield crns

	# Not many self dependencies, maybe move to another script in future
	def save_crn(self, _id, smiles, smiles_id):

		C_f = self.crns[_id][0]
		R_f = self.crns[_id][1]

		num_games_played = len(C_f[(smiles,0)].rewards)

		if self.current_game_number == (self.no_games - 1) or self.total_games == num_games_played:
			C_0, R_0 = self.temp_crns[smiles_id]
			greedy_training_states([C_f, R_f, C_0, R_0],
									smiles_id = smiles_id,
									max_depth = self.max_depth,
									game_no = self.current_game_number,
									FP_rad = 3,
									FPS_size = 16384,
									fileName = self.train_fid,
									verbose = self.verbose)
			mol_file = self.mols_fileName + "mol_{}.pkl".format(smiles_id)
			gc_gzip.disable()
			with gzip.open(mol_file, "wb") as fod:
				pickle.dump([smiles_id, smiles, C_f, R_f],
							io.BufferedWriter(fod),
							pickle.HIGHEST_PROTOCOL
							)
				gc_gzip.enable()

	# Has not many self dependencies, maybe move to another script in future
	def save_pathway(self, _id, pathway, target_cost, target_id, branching,
						buyable, save_for_training = False):

		if self.pathway_fileName:
			with open(self.pathway_fileName, "a+b") as fid:
				pickle.dump(pathway, fid, pickle.HIGHEST_PROTOCOL)

		c_branching, r_branching = branching
		c_branching = [str(c_branching[k]) for k in range(1,self.max_depth+1)]
		r_branching = [str(r_branching[k]) for k in range(1,self.max_depth+1)]
		branching = c_branching + r_branching
		branching = " ".join(branching)
		print_out = "{} {} {} {}\n".format(
					target_id,target_cost,int(buyable),branching
					)

		with open(self.outcomes_fileName, "a+") as fid:
			fid.write(print_out)

		if save_for_training:
			### Save pathway fps, cost
			pathway = self.pathways[_id]
			chemicals = pathway['chemical_nodes']
			reactions = pathway['reaction_nodes']
			target_smiles = pathway['target']
			smiles_id = pathway['smiles_id']

			C = self.crns[_id][0]
			R = self.crns[_id][1]
			chemicals = {key: C[key] for key in chemicals}
			reactions = {key: R[key] for key in reactions}

			greedy_training_states([chemicals, reactions, {}, {}],
									smiles_id = smiles_id,
									max_depth = self.max_depth,
									game_no = self.current_game_number,
									FP_rad = 3,
									FPS_size = 16384,
									fileName = self.train_fid,
									verbose = self.verbose)

	def update_tree(self, _id, save_crn = True, save_path = True):
		try:
			CC = self.crns[_id][0]
			RR = self.crns[_id][1]
			pathway = self.pathways[_id]

			chemicals = pathway['chemical_nodes']
			reactions = pathway['reaction_nodes']
			target_smiles = pathway['target']
			smiles_id = pathway['smiles_id']

			# Add in the penalties to the 'purchase price' so they get counted right in Mincost
			for key, C in chemicals.items():
				if C.retro_results == [] or (not len(C.incoming_reactions) and float(C.purchase_price) == -1.0):
					C.price(self.max_penalty)
					continue
				if key[1] == self.max_depth:
					C.price(self.depth_penalty)
					continue

			# Update costs / successes, visit counts
			Reset(chemicals,reactions)
			target_cost = MinCost((target_smiles, 0), self.max_depth, chemicals, reactions)

			buyable = True
			for chem_key in chemicals:
				CC[chem_key].visit_count += 1
				cost = chemicals[chem_key].cost
				if not np.isfinite(cost): # Covers single case when no templates apply to the target.
					target_cost = self.max_penalty
				try:
					CC[chem_key].rewards.append(cost)
				except:
					CC[chem_key].rewards = [cost]
				if len(chemicals[chem_key].incoming_reactions) == 0:
					# Check if leaf chemicals have finite ppg.
					if not (chemicals[chem_key].purchase_price == 0.0):
						buyable = False

			if buyable:
				self.successful_pathway_count += 1

			c_branching = {k: 0 for k in range(1,self.max_depth+1)}
			r_branching = {k: 0 for k in range(1,self.max_depth+1)}
			for reac_key in reactions:
				reac_smiles, depth1 = reac_key
				c_branching[depth1] += len(reac_smiles.split("."))
				r_branching[depth1] += 1
				RR[reac_key].visit_count += 1
				cost = reactions[reac_key].cost
				if np.isfinite(cost):
					try:
						RR[reac_key].rewards.append(cost)
					except:
						RR[reac_key].rewards = [cost]

			# Save chemicals and reactions
			if save_crn:
				self.save_crn(_id,target_smiles,smiles_id)
			if save_path:
				self.current_crns[smiles_id] = [smiles_id, target_smiles, CC, RR]
				self.pathway_count += 1
				self.save_pathway(_id,pathway,target_cost,smiles_id,[c_branching,r_branching],buyable)
		except:
			print "Error in update_tree:", traceback.format_exc()

	def coordinate(self):
		try:
			start_time = time.time()
			elapsed_time = time.time() - start_time
			next = 1
			finished = False
			while (elapsed_time < self.expansion_time) and self.waiting_for_results():

				if (int(elapsed_time)/10 == next):
					next += 1
					print "Worked for {}/{} s".format(int(elapsed_time*10)/10.0, self.expansion_time)
					print "... attempts {}\n... pathways {}".format(self.pathway_count,self.successful_pathway_count)

				try:
					for (_id, chem_smi, depth, precursors) in self.get_ready_result():
						children = self.add_reactants(_id,
													  chem_smi,
													  depth,
													  precursors
													  )
						self.active_chemicals[_id].remove((chem_smi,depth))
						if bool(children):
							if children == 'cyclic' or children == 'unexpandable':
								continue
							if (len(children) + self.pathway_status[_id][0] <= self.pathway_status[_id][2]):
								for kid in children: self.active_chemicals[_id].append(kid)
								_expand = self.expand_products(_id, children, self.rollout_branching)
								continue
						self.pathway_status[_id][1] = False

					for _id in range(self.nproc):

						no_worker = bool(self.idle[_id])
						is_pathway = bool(self.pathways[_id])
						no_results = self.results_queues[_id].empty()
						no_expansions = self.expansion_queues[_id].empty()
						is_pathway_dead = (not self.pathway_status[_id][1])
						check_dead = all([no_worker, is_pathway, no_results, no_expansions, is_pathway_dead])

						if check_dead:
							processed = [chem_dict.processed for chem_dict in self.pathways[_id]['chemical_nodes'].values()]
							if all(processed):
								self.update_tree(_id)
								self.pathways[_id] = 0
								self.active_chemicals[_id] = []
								self.pathways_queue.put(_id)
								#print "... put pathway (1) into pathways queue ... "
							elif (self.pathway_status[0] >= self.total_applied_templates) and (not self.active_chemicals[_id]):
								self.update_tree(_id)
								self.pathways[_id] = 0
								self.active_chemicals[_id] = []
								self.pathways_queue.put(_id)
								#print "... put pathway (2) into pathways queue ... "
							else:
								pass

						else:
							is_pathway = bool(self.pathways[_id])
							if is_pathway:
								processed = [chem_dict.processed for chem_dict in self.pathways[_id]['chemical_nodes'].values()]
								no_results = self.results_queues[_id].empty()
								no_expansions = self.expansion_queues[_id].empty()
								active_chemicals = (not self.active_chemicals[_id])
								check_delayed = all([no_results, no_expansions, active_chemicals])
								if check_delayed and processed:
									if all(processed):
										self.update_tree(_id)
										self.pathways[_id] = 0
										self.active_chemicals[_id] = []
										self.pathways_queue.put(_id)

					# Do not go further if we hit stopping condition.
					if finished:
						status = [(self.pathways[_id] == 0) for _id in range(self.nproc)]
						nothing_active = [(len(self.active_chemicals[_id]) == 0) for _id in range(self.nproc)]
						no_new_results = [self.results_queues[_id].empty() for _id in range(self.nproc)]
						no_new_expansions = [self.expansion_queues[_id].empty() for _id in range(self.nproc)]

						if all(status + nothing_active + no_new_results + no_new_expansions):
							break

						if not all(status) and all(no_new_results + no_new_expansions):
							for _id in range(self.nproc):
								if (not status[_id]):
									processed = [chem_dict.processed for chem_dict in self.pathways[_id]['chemical_nodes'].values()]
									if all(processed):
										self.update_tree(_id)
										self.pathways[_id] = 0
										self.active_chemicals[_id] = []
						continue

					for _id in self.get_pathway_result():

						a = time.time()
						if len(self.file_generator):
							calls, crn_data = self.file_generator.pop(0)
							if calls == self.group_size:
								finished = True
							#print len(self.file_generator), calls, crn_data[0], crn_data[1]
						else:
							finished = True
							self.pathways[_id] = 0
							break

						smiles_id, smiles, Chemicals, Reactions = crn_data

						#synthesis_attempts = len(Chemicals[(smiles,0)].rewards)
						#if synthesis_attempts >= self.total_games:
						#	continue

						if not self.current_game_number:
							#self.ResetCostEstimate(Chemicals, Reactions)
							C_0 = copy.deepcopy(Chemicals)
							R_0 = copy.deepcopy(Reactions)
							self.temp_crns[smiles_id] = [C_0, R_0]

						self.crns[_id] = [Chemicals, Reactions]
						self.current_crns[smiles_id] = [smiles_id, smiles, self.crns[_id][0],self.crns[_id][1]]

						leaves, pathway, expandable = self.Leaf_Generator(_id,smiles,smiles_id)
						self.pathways[_id] = pathway

						for (chem_key, chem_dict) in self.pathways[_id]['chemical_nodes'].items():
							if chem_key not in leaves: chem_dict.processed = True
						if expandable:
							self.pathway_status[_id] = [0, True, self.total_applied_templates]
							self.set_initial_target(_id,leaves)
						else:
							self.pathway_status[_id] = [0, False, self.total_applied_templates]

						elapsed_time = time.time() - start_time

				except Exception as E:
					print "... unspecified ERROR:", traceback.format_exc()
					elapsed_time = time.time() - start_time

			# Drain anything left in queues -- this shouldn't be necessary.
			#
			#
			#
			for _id in self.get_pathway_result():
				continue
			for leftovers in self.get_ready_result():
				continue
			for _id in range(self.nproc):
				self.pathways[_id] = 0

		except:
			print "Error in coordinate:", traceback.format_exc()
			sys.exit(1)

	def work(self, i):

		if self.precursor_prioritization == gc.mincost and bool(self.model_weights):
			from makeit.prioritization.precursors.mincost import MinCostPrecursorPrioritizer
			model = MinCostPrecursorPrioritizer()

			# On occasion we get a crash when trying to load model -- multiproc
			# issue with locks not being handled (released) properly by theano.
			attempt = 0
			while attempt < 10:
				try:
					model.load_model(datapath=self.model_weights + '/')
					break
				except:
					attempt += 1
			if attempt == 10:
				print "Failed to load n.n. cost model ... exiting."
				sys.exit(1)

		while True:
			# If done, stop
			if self.done.value:
				break

			# If paused, wait and check again
			if self.paused.value:
				time.sleep(1)
				continue

			# Grab something off the queue
			try:
				self.idle[i] = False
				(jj, smiles, depth, branching, outcomes) = self.expansion_queues[i].get(timeout=0.1)  # short timeout
				prioritizers = (self.precursor_prioritization,
								self.template_prioritization
								)

				#print jj, smiles, depth, branching, outcomes

				if not outcomes:

					if self.precursor_prioritization == gc.mincost:
						prioritizers = (gc.relevance_precursor,
										self.template_prioritization
										)
					outcomes = self.retroTransformer.get_outcomes(
						smiles,
						self.mincount,
						prioritizers,
						depth = depth,
                        template_count = self.template_count,
                        mode = self.precursor_score_mode,
                        max_cum_prob = self.max_cum_template_prob
                        )

				if self.precursor_prioritization == gc.mincost:
					for precursor in outcomes.precursors:
						if bool(self.model_weights):
							precursor.retroscore = 1.0 + sum([abs(model.get_score_from_smiles(smile,depth+1)) for smile in precursor.smiles_list])
						else:
							precursor.retroscore = 1.0 + sum([random.randint(0,self.max_penalty) for smile in precursor.smiles_list])

				reaction_precursors = outcomes.return_top(n=self.rollout_branching)
				self.results_queues[i].put((jj, smiles, depth, [reaction_precursors, outcomes]))

			except VanillaQueue.Empty:
				pass

			except Exception as e:
				print traceback.format_exc()

			self.idle[i] = True

	def add_reactants(self, _id, chem_smi, depth, rxn_precursors):
		try:
			if (chem_smi, depth) not in self.pathways[_id]['chemical_nodes']:
				print "ERROR: Target not in pathway. Exiting ... "
				sys.exit(1)

			C = self.crns[_id][0]
			R = self.crns[_id][1]
			pathway = self.pathways[_id]
			path_status = self.pathway_status[_id]
			active_chemicals = self.active_chemicals[_id]

			precursors, outcomes = rxn_precursors
			pathway['chemical_nodes'][(chem_smi,depth)].processed = True
			# If no templates applied, do not go further, chemical not makeable.
			if not precursors:
				C[chem_smi,depth].retro_results = []
				pathway['chemical_nodes'][chem_smi,depth].retro_results = []
				return 'unexpandable'
			C[chem_smi,depth].retro_results = outcomes

			scores_list = []
			for result in precursors:
				reactants = result['smiles_split']
				retroscore = result['score']
				template_action = result['tforms']
				template_probability = result['template_score']
				# Reject cyclic templates as 'illegal moves'.
				cyclic_template = False
				for q,smi in enumerate(reactants):
					if smi in pathway['chemicals']:
						reactant_smile_key = sorted([(rchem_smi,rdepth) for (rchem_smi,rdepth) in pathway['chemical_nodes'] if (rchem_smi == smi)], key=lambda x: x[1])[0]
						if not (pathway['chemical_nodes'][reactant_smile_key].purchase_price >= 0):
							last_reactant_cost = pathway['chemical_nodes'][reactant_smile_key].cost
							if (last_reactant_cost >= self.max_penalty) or (last_reactant_cost == -1):
								cyclic_template = True
								break
				if cyclic_template:
					continue
				scores_list.append([retroscore,reactants,template_probability,template_action])

			if not scores_list:
				C[chem_smi,depth].retro_results = []
				pathway['chemical_nodes'][chem_smi,depth].retro_results = []
				return 'unexpandable'

			results = sorted(scores_list,
							key=lambda x: (x[0], sum([len(xx) for xx in x[1]])))

			# Epsilon-greedy selection
			if (random.random() < self.epsilon) and len(results) > 0:
				random.shuffle(results)

			pathway['chemical_nodes'][chem_smi,depth].retro_results = results

			for p,result in enumerate(results):
				react_cost, reactants, template_prob, template_no = result
				if isinstance(reactants, list):
					rxn_smi = ".".join(reactants)
				else:
					rxn_smi = reactants

				C[chem_smi,depth].add_incoming_reaction(rxn_smi,(template_no,template_prob))
				if (rxn_smi,depth+1) not in R:
					R[rxn_smi,depth+1] = Reaction(rxn_smi,depth+1)
				R[rxn_smi,depth+1].cost_estimate = react_cost
				R[rxn_smi,depth+1].add_outgoing_chemical(chem_smi,(template_no,template_prob))

				if p == 0:
					children = []
					pathway['chemical_nodes'][(chem_smi,depth)].add_incoming_reaction(rxn_smi,(template_no,template_prob))
					pathway['chemical_nodes'][(chem_smi,depth)].retro_results = result
					pathway['reaction_nodes'][(rxn_smi,depth+1)] = Reaction(rxn_smi,depth+1)
					pathway['reaction_nodes'][(rxn_smi,depth+1)].add_outgoing_chemical(chem_smi,(template_no,template_prob))

				for q,smi in enumerate(reactants):
					R[rxn_smi,depth+1].add_incoming_chemical(smi)
					if (smi,depth+1) not in C:
						C[smi,depth+1] = Chemical(smi,depth+1)
						C[smi,depth+1].price(self.get_price(smi))
					C[smi,depth+1].add_outgoing_reaction(rxn_smi)

					if p == 0:
						if (smi,depth+1) in children: continue
						children.append((smi,depth+1))
						pathway['reaction_nodes'][(rxn_smi,depth+1)].add_incoming_chemical(smi)
						if (smi,depth+1) not in pathway['chemical_nodes']:
							pathway['chemical_nodes'][smi,depth+1] = Chemical(smi,depth+1)
							ppg = C[smi,depth+1].purchase_price #self.get_price(smi)
							pathway['chemical_nodes'][smi,depth+1].price(ppg)
							if (ppg >= 0.0) or ((depth + 1) == self.max_depth) or (not path_status[1]):
								pathway['chemical_nodes'][smi,depth+1].processed = True
							if ((depth + 1) == self.max_depth) and (not ppg >= 0.0):
								path_status[1] = False

			#################################
			if children and (depth < self.max_depth) and path_status[1]:
				#not cyclic_template
				return children
			else:
				#print "Warning (ii): Nothing left to expand.", cyclic_template
				for smi in reactants:
					try:
						if (not pathway['chemical_nodes'][smi,depth+1].processed):
							pathway['chemical_nodes'][smi,depth+1].processed = True
					except:
						pass
				return False

		except Exception as E:
			print "Error in add_reactants:", traceback.format_exc()
			print self.pathways[_id]['chemicals']
			for key, c in self.pathways[_id]['chemical_nodes'].items():
				print key, type(c.retro_results)

	def expand_products(self, _id, children, branching):
		try:
			C = self.crns[_id][0]
			R = self.crns[_id][1]
			pathway = self.pathways[_id]
			path_status = self.pathway_status[_id]
			active_chemicals = self.active_chemicals[_id]

			synthetic_expansion_candidates = 0
			for (chem_smi, depth) in children:
				if depth >= self.max_depth:
					active_chemicals.remove((chem_smi,depth))
					path_status[1] = False
					continue

				if (chem_smi,depth) not in C:
					C[chem_smi,depth] = Chemical(chem_smi,depth)
					C[chem_smi,depth].price(self.get_price(chem_smi))

				ppg = C[chem_smi,depth].purchase_price
				if chem_smi in pathway['chemicals']:
					if not (ppg >= 0.0):
						active_chemicals.remove((chem_smi,depth))
						path_status[1] = False
						continue

				pathway['chemicals'].add(chem_smi)
				if (chem_smi,depth) not in pathway['chemical_nodes']:
					pathway['chemical_nodes'][chem_smi,depth] = Chemical(chem_smi,depth)
				pathway['chemical_nodes'][chem_smi,depth].price(ppg)

				if ppg >= 0:
					pathway['chemical_nodes'][(chem_smi,depth)].processed = True
					active_chemicals.remove((chem_smi,depth))
					continue

				if not (path_status[0] < path_status[2]):
					pathway['chemical_nodes'][(chem_smi,depth)].processed = True
					active_chemicals.remove((chem_smi,depth))
					path_status[1] = False
					continue

				synthetic_expansion_candidates += 1
				path_status[0] += 1
				self.expand(_id,chem_smi,depth,branching)

			return synthetic_expansion_candidates

		except Exception as e:
			print "Error in expand_products:", traceback.format_exc()

	# Has not many self dependencies, maybe move to another script in future
	# Should anyways so UCT, etc may be substituted in.
	# Leaf_Generator may also be good to put into this class
	def epsilon_greedy(self, _id, product_key, pathway):
		try:
			C = self.crns[_id][0]
			R = self.crns[_id][1]
			incoming_reactions = C[product_key].incoming_reactions

			rxn_scores = []
			for (rxn_smiles, depth1, template_details) in incoming_reactions:
				template_id, template_probability = template_details
				reactant_smiles = rxn_smiles.split(".")
				cyclic_path = False
				for reactant_smi in reactant_smiles:
					if C[reactant_smi,depth1].purchase_price >= 0.0:
						continue
					if reactant_smi in pathway['chemicals']:
						cyclic_path = True
						break
				if cyclic_path:
					continue

				chemical_ppg = [C[react_smi,depth1].purchase_price for react_smi in reactant_smiles]
				accumulated_rewards = R[rxn_smiles,depth1].rewards
				if all([x > -1 for x in chemical_ppg]):
					rxn_score = 1.0 + sum(chemical_ppg)
				elif (self.precursor_prioritization == gc.mincost) and len(accumulated_rewards):
					rxn_score = np.mean(accumulated_rewards)
				else:
					rxn_score = R[rxn_smiles,depth1].cost_estimate
					if not np.isfinite(rxn_score):
						rxn_score = 1.0
						for smile in reactant_smiles:
							if C[smile,depth1].purchase_price >= 0.0:
								# Purchase prices are checked when node added to tree
								rxn_score += C[smile,depth1].purchase_price
							elif len(C[smile,depth1].rewards):
								rxn_score += np.mean(C[smile,depth1].rewards)
							elif np.isfinite(abs(C[smile,depth1].cost_estimate)):
								rxn_score += abs(C[smile,depth1].cost_estimate)
							else:
								if not bool(self.model_weights):
									C[smile,depth1].cost_estimate = random.randint(0,self.max_penalty)
								else:
									C[smile,depth1].cost_estimate = abs(self.model.get_score_from_smiles(smile,depth1))
								rxn_score += C[smile,depth1].cost_estimate

				rxn_scores.append([rxn_score,rxn_smiles,depth1,template_details])

			if len(rxn_scores) == 0:
				return [1, []]

			if len(rxn_scores) == 1:
				selected_reactants = rxn_scores[0]
			else:
				sorted_rxn_scores = sorted(rxn_scores, key=lambda x: x[0])
				if random.random() < self.epsilon:
					selected_reactants = random.choice(sorted_rxn_scores)
				else:
					best_rxn_score = sorted_rxn_scores[0]
					try:
						selected_reactants = random.choice([rxn for rxn in sorted_rxn_scores if rxn[0] == best_rxn_score[0]])
					except:
						selected_reactants = rxn_scores[0]

			return [0, selected_reactants]

		except:
			print "Greedy", traceback.format_exc()

	def UCB(self, _id, product_key, pathway):
		try:
			C = self.crns[_id][0]
			R = self.crns[_id][1]

			product_visits = C[product_key].visit_count
			incoming_reactions = C[product_key].incoming_reactions

			rxn_scores = []
			for (rxn_smiles, depth1, template_details) in incoming_reactions:
				reactant_smiles = rxn_smiles.split(".")

				cyclic_path = False
				for reactant_smi in reactant_smiles:
					if C[reactant_smi,depth1].purchase_price >= 0.0:
						continue
					if reactant_smi in pathway['chemicals']:
						cyclic_path = True
						break
				if cyclic_path:
					continue

				rxn_visits = float(R[rxn_smiles,depth1].visit_count)
				if rxn_visits < 1e-3: #If not visited, give max score.
					rxn_score = 100.0
				elif product_visits < 1e-3:
					rxn_score = 100.0
				else:
					bonus = 2.0 * self.c_exploration * np.sqrt(2.0 * np.log(product_visits) / rxn_visits)
					#UCT-like: #bonus = c_exploration * np.sqrt(product_visits)/(1.0 + rxn_visits)
					#ave_rxn_cost = sum([success for success in self.Reactions[rxn_smiles,depth1].successes]) / float(len(self.Reactions[rxn_smiles,depth1].successes))
					ave_rxn_cost = sum([success * cost for (success,cost) in zip(R[rxn_smiles,depth1].successes, R[rxn_smiles,depth1].rewards)]) / float(len(R[rxn_smiles,depth1].successes))
					# Scale cost so 'largest' ~ mincost.
					#maximum_rxn_cost = len(reactant_smiles) * self.depth_penalty
					#scaled_rxn_cost = (maximum_rxn_cost - ave_rxn_cost) / maximum_rxn_cost
					#rxn_score = scaled_rxn_cost + bonus
					rxn_score = ave_rxn_cost + bonus
				rxn_scores.append([rxn_score,rxn_smiles,depth1,template_details])

			if len(rxn_scores) == 0:
				return [1, []]

			if len(rxn_scores) == 1:
				selected_reactants = rxn_scores[0]
			else:
				sorted_rxn_scores = sorted(rxn_scores, key=lambda x: x[0])[::-1]
				best_rxn_score = sorted_rxn_scores[0]
				selected_reactants = random.choice([rxn for rxn in sorted_rxn_scores if rxn[0] == best_rxn_score[0]])
				#print best_rxn_score[0], [score[0] for score in sorted_rxn_scores]

			return [0, selected_reactants]

		except Exception as E:
			print "UCB", traceback.format_exc()

	def TreePolicy(self):
		if self.tree_policy == "epsilon-greedy":
			self.tree_policy = self.epsilon_greedy
		elif self.tree_policy == "mcts":
			self.tree_policy = self.UCB
		else:
			print "Currently must select tree-policy from: epsilon-greedy  \
					or mcts. Exiting. "
			sys.exit(1)

	def Leaf_Generator(self, _id, smiles, smiles_id):
		try:
			C = self.crns[_id][0]
			R = self.crns[_id][1]
			current_pathway = {'chemicals': set(),
							   'chemical_nodes': {},
							   'reaction_nodes': {},
							   'target': smiles,
		   		   			   'smiles_id': smiles_id
							   }
			# If starting fresh
			if (smiles,0) not in C:
				return ([[(smiles,0)], current_pathway, True])

			initial_state = (1.0,[smiles, 0])
			queue = VanillaQueue.PriorityQueue()
			queue.put(initial_state)

			cyclic_path = False
			possible_leaves = []
			while not queue.empty():
				chem_score, _data = queue.get()
				rxn_smiles, depth = _data
				rxn_smiles = rxn_smiles.split(".")

				for chem_smi in rxn_smiles:
					CC = C[chem_smi,depth]
					#current_cost = CC.cost
					retro_results = CC.retro_results
					purchase_price = CC.purchase_price
					#product_visits = CC.visit_count
					incoming_reactions = CC.incoming_reactions

					if (chem_smi,depth) not in current_pathway['chemical_nodes']:
						current_pathway['chemical_nodes'][chem_smi,depth] = Chemical(chem_smi,depth)

					# Do not put buyables in the queue.
					if purchase_price >= 0:
						current_pathway['chemical_nodes'][chem_smi,depth].price(purchase_price)
						continue

					if retro_results == None and depth < self.max_depth:
						# Not expanded yet so a possible leaf.
						possible_leaves.append((chem_smi,depth))
						continue

					# Update chemical node details.
					current_pathway['chemicals'].add(chem_smi)
					cyclic_path, selected_reactants = self.tree_policy(
															_id,
															(chem_smi,depth),
															current_pathway
															)
					if cyclic_path:
						continue
					score, rxn_smiles, depth1, template_details = selected_reactants

					## Add nodes to pathway
					current_pathway['chemical_nodes'][chem_smi,depth].add_incoming_reaction(rxn_smiles,template_details)
					current_pathway['chemical_nodes'][chem_smi,depth].retro_results = selected_reactants
					current_pathway['reaction_nodes'][rxn_smiles,depth1] = Reaction(rxn_smiles,depth1)
					current_pathway['reaction_nodes'][rxn_smiles,depth1].add_outgoing_chemical(chem_smi,template_details)

					for smi in rxn_smiles.split("."):
						current_pathway['reaction_nodes'][rxn_smiles,depth1].add_incoming_chemical(smi)
						if (smi,depth1) not in current_pathway['chemical_nodes']:
							current_pathway['chemical_nodes'][smi,depth1] = Chemical(smi,depth1)
							current_pathway['chemical_nodes'][smi,depth1].price(C[smi,depth1].purchase_price)

					reaction_state = (score, [rxn_smiles,depth1])
					queue.put(reaction_state)

			return ([possible_leaves, current_pathway, bool(len(possible_leaves))])

		except Exception as E:
			print traceback.format_exc()

	def build_tree(self):
		start_time = time.time()
		self.running = True
		self.prepare()

		# Initialize tree policy method (epsilon-greedy or UCB)
		self.TreePolicy()

		# Load current cost network for use in selection, if we don't have an estimate.
		if bool(self.model_weights):
			self.model = self.estimate_current_cost()

		# Play games for batches of chemicals.
		self.group_size = 24

		# Open file for saving training states
		self.train_fid = open(self.training_states_fileName, "a+")

		big_a = time.time()
		for group_ID, targets in enumerate(self.load_crn(self.target_chemicals)):

			if (time.time()-start_time > self.expansion_time):
				break

			# Try to play X games
			self.temp_crns = {}
			for attempt in range(self.no_games):

				small_a = time.time()
				self.pathway_count = 0
				self.successful_pathway_count = 0
				self.current_game_number = attempt
				self.current_crns = {}

				#if self.verbose:
				print "Group {} | Game {} / {} | Epsilon {}".format(group_ID, attempt + 1, self.no_games, self.epsilon)
				start_a = time.time()

				# Load (C,R) generator.
				self.file_generator = targets
				self.group_size = len(self.file_generator)

				launched = 0
				while launched < self.nproc:
					if len(self.file_generator):
						calls, crn_data = self.file_generator.pop(0)
					else:
						break
					smiles_id, smiles, Chemicals, Reactions = crn_data

					if not self.current_game_number:
						#self.ResetCostEstimate(Chemicals, Reactions)
						self.temp_crns[smiles_id] = [copy.deepcopy(Chemicals),copy.deepcopy(Reactions)]

					# Do not run if we already have target number of rewards for target.
					if (smiles,0) in Chemicals:
						synthesis_attempts = len(Chemicals[(smiles,0)].rewards)
						if synthesis_attempts >= self.total_games:
							continue

					self.crns[launched] = [Chemicals, Reactions]#, copy.deepcopy(Chemicals), copy.deepcopy(Reactions)]
					self.current_crns[smiles_id] = [smiles_id, smiles, self.crns[launched][0],self.crns[launched][1]]

					leaves, pathway, expandable = self.Leaf_Generator(
													launched,smiles,smiles_id
													)

					#if self.verbose:
					#	print "(a) Launched so far", launched, "current id", smiles_id, "..."

					self.pathways[launched] = pathway
					for (chem_key, chem_dict) in self.pathways[launched]['chemical_nodes'].items():
						if chem_key not in leaves:
							chem_dict.processed = True
					if expandable:
						_status = [0, True, self.total_applied_templates]
						self.pathway_status[launched] = _status
						self.set_initial_target(launched,leaves)
					else:
						_status = [0, False, self.total_applied_templates]
						self.pathway_status[launched] = _status
					launched += 1

				'''
				Wait 1-sec before proceeding. Mainly here for cases when the
				group size/workers is 1 and coordinate() finishes before work
				finishes.
				'''
				time.sleep(1)

				# Coordinate workers.
				if launched:
					self.coordinate()

				if self.verbose:
					print "... it took {} s to expand group {}".format(
						time.time()-small_a, group_ID
						)

				# Save group_IDs that ran -> in case we reach max wall_time and need to resubmit
				with open(self.logger_fileName, "a+") as lid:
					lid.write("{} {} {}\n".format(
								group_ID,self.pathway_count,
								self.successful_pathway_count)
								)

				# Precautionary drain the queues -> there should not be anything left.
				for _id in self.get_pathway_result():
					continue
				for leftovers in self.get_ready_result():
					continue
				for _id in range(self.nproc):
					self.pathways[_id] = 0


				### SHOULD BE REDUNDANT ...


				# Do not add targets if we have the desired number of attempts
				if not len(self.current_crns.values()):
					break

				counter = 1
				targets = []
				for x in self.current_crns.values():
					_id, smiles, C, R = x
					if len(C[smiles,0].rewards) < self.total_games:
						targets.append([counter,x])
						counter += 1
				if not len(targets):
					break

		working_time = time.time()-big_a
		print "Finished working. It took {} to play {} games".format(
																working_time,
																self.no_games
																)
		self.train_fid.close()
		self.stop()

	def reset(self):
		self.manager = Manager()
		self.done = self.manager.Value('i', 0)
		self.paused = self.manager.Value('i', 0)
		self.idle = self.manager.list()
		self.results_queue = Queue()
		self.workers = []
		self.coordinator = None
		self.running = False

		## Queues
		self.crns = [0 for i in range(self.nproc)]
		self.pathways = [0 for i in range(self.nproc)]
		self.pathways_queue = Queue()
		self.pathway_status = [[0,True,self.total_applied_templates] for i in range(self.nproc)]
		self.sampled_pathways = []
		self.pathway_count = 0
		self.successful_pathway_count = 0

		for i in range(self.nproc):
			self.idle.append(True)
		if self.nproc != 1:
			self.expansion_queues = [Queue() for i in range(self.nproc)]
			self.results_queues   = [Queue() for i in range(self.nproc)]
		else:
			self.expansion_queues = [Queue()]
			self.results_queues   = [Queue()]
		self.active_chemicals = [[] for x in range(self.nproc)]

	def get_buyable_paths(self,
							target_chemicals,
							replica = 0,
							max_depth = 10,
							no_games = 1,
							total_games = 1,
							expansion_time = 300,
							expansion_branching = 1,
							rollout_branching = 1,
							total_applied_templates = 1000,
							template_prioritization=gc.relevance,
				 			precursor_prioritization=gc.heuristic,
				 			model_weights = '', tree_policy = 'epsilon-greedy',
				 			mincount=25, chiral=True, epsilon = 0.0,
							c_exploration = 0.0, template_count = 50,
							precursor_score_mode=gc.max,
                          	max_cum_template_prob = 0.995):
		self.target_chemicals = target_chemicals
		self.replica = replica
		self.mincount = mincount
		self.max_depth = max_depth
		self.no_games = no_games
		self.total_games = total_games
		self.tree_policy = tree_policy
		self.c_exploration = c_exploration
		self.epsilon = epsilon
		if self.precursor_prioritization == 'random':
			self.epsilon = 1.0
		self.expansion_time = expansion_time
		self.expansion_branching = expansion_branching
		self.rollout_branching = rollout_branching
		self.total_applied_templates = total_applied_templates
		self.template_prioritization = template_prioritization
		self.precursor_prioritization = precursor_prioritization
		self.precursor_score_mode = precursor_score_mode
		self.model_weights = model_weights
		self.template_count = template_count
		self.max_cum_template_prob = max_cum_template_prob

		self.depth_penalty = score_max_depth()
		self.max_penalty = score_no_templates()

		self.manager = Manager()
		# specificly for python multiprocessing
		self.done = self.manager.Value('i', 0)
		self.paused = self.manager.Value('i', 0)
		# Keep track of idle workers
		self.idle = self.manager.list()
		self.results_queue = Queue()
		self.workers = []
		self.coordinator = None
		self.running = False
		## Queues
		self.crns = [0 for i in range(self.nproc)]
		self.pathways = [0 for i in range(self.nproc)]
		self.pathways_queue = Queue()
		self.pathway_status = [[0,True,self.total_applied_templates] for i in range(self.nproc)]
		self.sampled_pathways = []
		self.pathway_count = 0
		self.successful_pathway_count = 0

		for i in range(self.nproc):
			self.idle.append(True)
		if self.nproc != 1:
			self.expansion_queues = [Queue() for i in range(self.nproc)]
			self.results_queues   = [Queue() for i in range(self.nproc)]
		else:
			self.expansion_queues = [Queue()]
			self.results_queues   = [Queue()]
		self.active_chemicals = [[] for x in range(self.nproc)]

		return self.build_tree()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--delay_time', type = float, default = 20.0 * 60.0,
						help = 'Terminate X seconds before hitting wall-time \
								(default 20 mins)'
								)
	parser.add_argument('--wall_time', type = float, default = 12 * 3600,
						help = 'Local wall-time (seconds). Default: 12 hours')
	parser.add_argument('--nodes', type = int, default = 1,
						help = 'Number of nodes. Default = 1.')
	parser.add_argument('--cores', type = int, default = psutil.cpu_count(),
						help = 'Number of cpu cores. Default = All available')
	parser.add_argument('--replica', type = int, default = 0,
						help = 'If nodes > 1, must specify which "replica" \
								to label this script'
								)
	parser.add_argument('--epsilon', type = float, default = 0.0,
						help = 'Exploration epsilon (may select from 0 to 1). \
								Default: 0.0'
								)
	parser.add_argument('--c_exploration', type = float, default = 0.0,
						help = 'Exploration in UCT/UCB (may select from 0 to 1). \
								Default: 0.3'
								)
	parser.add_argument('--policy', type = str, default = 'heuristic',
						help = 'Select from: heuristic (default), relevance, cost')
	parser.add_argument('--tree_policy', type = str, default = 'epsilon-greedy',
						help = 'Select from: epsilon-greedy (D) or UCB')
	parser.add_argument('--truncate_branching', type = int, default = 50,
						help = 'Limit the number of reaction presursors (D=50)')
	parser.add_argument('--model_weights', type = str, default = '',
						help = 'Path to neural network weights. Default is None \
								If policy is cost and model weights not \
								specified, makeit model will be used.'
								)
	parser.add_argument('--smiles', type = str, default = './states/smiles.txt',
						help = 'Select from: heuristic, relevance, cost (default)')
	parser.add_argument('--verbose', type = bool, default = False,
						help = 'Print results from synthesis attempt. \
								Default = False'
								)
	parser.add_argument('--games_per_target', type = int, default = 1,
						help = 'Number of synthesis attempts (games) per target \
								Default = 1.'
								)
	parser.add_argument('--max_games', type = int, default = 1,
						help = 'Used for restarting if  \
								games_per_target < max_games \
								at the wall-time. Default = 1.'
								)
	parser.add_argument('--data_path', type = str, default = 'output',
						help = 'Directory where results should be saved. \
						Default is cwd/output'
								)
	args = parser.parse_args()

	nodes 			= int(args.nodes)
	cores 			= min(24, int(args.cores))
	replica 		= int(args.replica)

	wall_time 		= float(args.wall_time)
	delay_time		= float(args.delay_time)

	policy 			= str(args.policy)
	tree_policy  	= str(args.tree_policy)
	model_weights 	= str(args.model_weights)
	lim_branching 	= int(args.truncate_branching)

	epsilon 		= min(1, float(args.epsilon))
	c_exploration 	= min(1, float(args.c_exploration))

	smiles 			= str(args.smiles)
	verbose			= str(args.verbose)

	games_per_target 	= int(args.games_per_target)  # Respect the wall-time
	max_games 			= int(args.max_games)

	data_path 			= str(args.data_path)

	assert os.path.isfile(smiles), "Could not find target \
									smiles at {}".format(smiles)

	if policy == "heuristic":
		policy = gc.heuristic
		model_weights = ''
	elif policy == "relevance":
		policy = gc.relevance_precursor
		model_weights = ''
	else:
		policy = gc.mincost
		if bool(model_weights):
			assert os.path.isdir(model_weights), "Weight dir does not exist"

	training_samples = smilesLoader(smiles, verbose = True).load(nodes, replica)

	# Create directories for output data streams.
	for data_dir in ['crn', 'logs', 'pathways', 'states']:
		joined_path = os.path.join(data_path, data_dir)
		if not os.path.isdir(joined_path):
			os.makedirs(joined_path)

	# Set up files for data streams
	stringer = (data_path,policy,epsilon,replica)
	mols_fileName = "{}/crn/".format(data_path)
	logger_fileName = "{}/logs/logger_{}_{}_{}.txt".format(*stringer)
	pathway_fileName = "{}/pathways/paths_{}_{}_{}.pkl".format(*stringer)
	outcomes_fileName = "{}/logs/outcomes_{}_{}_{}.txt".format(*stringer)
	training_states_fileName = "{}/states/states_{}_{}_{}.pkl".format(*stringer)

	print "There are {} cores available ... ".format(cores)
	Tree = TreeGenerator(nproc = cores,
						verbose = verbose,
						filePath = data_path,
						load_transformer = True,
						mols_fileName = mols_fileName,
						logger_fileName = logger_fileName,
						pathway_fileName = pathway_fileName,
						outcomes_fileName  = outcomes_fileName,
						training_states_fileName = training_states_fileName
						)

	a = time.time()
	result = Tree.get_buyable_paths(training_samples,
									replica = replica,
									no_games = games_per_target,
									total_games = max_games,
									tree_policy = tree_policy,
									epsilon = epsilon,
									c_exploration = c_exploration,
									expansion_time = wall_time,
									precursor_prioritization = policy,
									model_weights = model_weights,
									rollout_branching = lim_branching,
									)
	print "It took {} s to run {} games, {} mols".format(
		time.time()-a, games_per_target,
		games_per_target * len(training_samples)
		)
