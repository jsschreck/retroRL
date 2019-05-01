from makeit.retrosynthetic.transformer import RetroTransformer
from makeit.utilities.buyable.pricer import Pricer
from multiprocessing import Process, Manager, Queue
from celery.result import allow_join_result
from pymongo import MongoClient

from makeit.mcts.cost import Reset, score_max_depth, score_no_templates, MinCost, BuyablePathwayCount
from makeit.mcts.misc import get_feature_vec, save_sparse_tree
from makeit.mcts.misc import value_network_training_states
from makeit.mcts.nodes import Chemical, Reaction

import makeit.global_config as gc
import Queue as VanillaQueue
import multiprocessing as mp
import cPickle as pickle 
import numpy as np
import traceback
import itertools
import argparse
import psutil
import random
import time 
import gzip 
import sys
import os

NCPUS = psutil.cpu_count()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['openmp'] = 'True'
os.environ['MKL_NUM_THREADS'] = '{}'.format(NCPUS)
os.environ['GOTO_NUM_THREADS'] = '{}'.format(NCPUS)
os.environ['OMP_NUM_THREADS'] = '{}'.format(NCPUS)

class MCTS:

	def __init__(self, retroTransformer=None, pricer=None, 
				 max_branching=50, total_applied_templates = 1000,
				 max_depth=10, celery=False, 
				 nproc=8, mincount=25, chiral=True, template_prioritization=gc.relevance, 
				 precursor_prioritization=gc.heuristic, max_ppg=100, 
				 mincount_chiral=10):

		self.celery = celery
		self.mincount = mincount
		self.mincount_chiral = mincount_chiral
		self.max_depth = max_depth  
		self.max_branching = max_branching
		self.total_applied_templates = total_applied_templates
		
		self.template_prioritization = template_prioritization
		self.precursor_prioritization = precursor_prioritization
		self.nproc = nproc
		self.chiral = chiral
		self.max_cum_template_prob = 1

		## Pricer
		if pricer:
			self.pricer = pricer
		else:
			self.pricer = Pricer()
		self.pricer.load(max_ppg=max_ppg)

		self.reset()

		## Load transformer 
		'''
		try:
			from makeit.utilities.io import model_loader
			if not self.celery:
				if retroTransformer:
					self.retroTransformer = retroTransformer
				else:
					self.retroTransformer = model_loader.load_Retro_Transformer(mincount=self.mincount,
			                                                            		mincount_chiral=self.mincount_chiral,
			                                                            		chiral=self.chiral)
		
		except:
		'''
		# model_loader tries to load mpl, don't have/want it on the cluster ... 
		# classical load then. 
		
		db_client = MongoClient(gc.MONGO['path'], gc.MONGO['id'], connect=gc.MONGO['connect'])
		TEMPLATE_DB = db_client[gc.RETRO_TRANSFORMS_CHIRAL['database']][gc.RETRO_TRANSFORMS_CHIRAL['collection']]
		self.retroTransformer = RetroTransformer(mincount=self.mincount, mincount_chiral=self.mincount_chiral, TEMPLATE_DB=TEMPLATE_DB)
		self.retroTransformer.chiral = self.chiral
		
		
		home = os.path.expanduser('~')
		if home.split("/")[1] == "rigel": home = "/rigel/cheme/users/jss2278/chemical_networks"
		transformer_filepath = home + "/Make-It/makeit/data/"
		if os.path.isfile(transformer_filepath+"chiral_templates.pickle"):
			self.retroTransformer.load_from_file(True, "chiral_templates.pickle", chiral=self.chiral, rxns=True, file_path=transformer_filepath)
		else:
			self.retroTransformer.dump_to_file(True, "chiral_templates.pickle", chiral=self.chiral, file_path=transformer_filepath)

		if self.celery:
			def expand(smiles, chem_id, queue_depth, branching):
				# Chiral transformation or heuristic prioritization requires
				# same database
				if self.chiral or self.template_prioritization == gc.relevance:
					self.pending_results.append(tb_c_worker.get_top_precursors.apply_async(
					    args=(smiles, self.template_prioritization,
					          self.precursor_prioritization),
					    kwargs={'mincount': self.mincount,
					            'max_branching': self.max_branching,
					            'template_count': self.template_count,
					            'mode': self.precursor_score_mode,
					            'max_cum_prob':self.max_cum_template_prob},
					    # Prioritize higher depths: Depth first search.
					    priority=int(depth),
					    queue=self.private_worker_queue,
					))
				else:
					self.pending_results.append(tb_worker.get_top_precursors.apply_async(
						args=(smiles, self.template_prioritization,
						      self.precursor_prioritization),
						kwargs={'mincount': self.mincount,
						        'max_branching': self.max_branching,
						        'template_count': self.template_count,
						        'mode': self.precursor_score_mode,
						        'max_cum_prob':self.max_cum_template_prob},
						# Prioritize higher depths: Depth first search.
						priority=int(depth),
						queue=self.private_worker_queue,
					))
		else:
			def expand(_id, chem_smi, depth, branching):
				self.expansion_queues[_id].put((_id, chem_smi, depth, branching))
		self.expand = expand

		# Define method to start up parallelization.
		if self.celery:
			def prepare():
				if self.chiral:
					self.private_worker_queue = tb_c_worker.reserve_worker_pool.delay().get(timeout=5)
				else:
					self.private_worker_queue = tb_worker.reserve_worker_pool.delay().get(timeout=5)
		else:
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
		if self.celery:
			def waiting_for_results():
				# update
				time.sleep(1)
				return self.pending_results != [] or self.is_ready != []
		else:
			def waiting_for_results():
				waiting = [expansion_queue.empty()
				           for expansion_queue in self.expansion_queues]
				for results_queue in self.results_queues:
					waiting.append(results_queue.empty())
				waiting += self.idle
				return (not all(waiting))
		self.waiting_for_results = waiting_for_results

		# Define method to get a processed result.
		if self.celery:
			def get_ready_result():
				#Update which processes are ready
				self.is_ready = [i for (i, res) in enumerate(self.pending_results) if res.ready()]
				for i in self.is_ready:
					(smiles, precursors) = self.pending_results[i].get(timeout=0.25)
					self.pending_results[i].forget()
					_id = self.chem_to_id[smiles]
					yield (_id, smiles, precursors)
				self.pending_results = [res for (i, res) in enumerate(
					self.pending_results) if i not in self.is_ready]
		else:
			def get_ready_result():
				for results_queue in self.results_queues:
					while not results_queue.empty():
						yield results_queue.get(timeout=0.25)
		self.get_ready_result = get_ready_result

		# Define method to get a signal to start a new attempt.
		if self.celery:
			def get_pathway_result():
				#Update which processes are ready
				self.is_ready = [i for (i, res) in enumerate(self.pending_results) if res.ready()]
				for i in self.is_ready:
					(smiles, precursors) = self.pending_results[i].get(timeout=0.25)
					self.pending_results[i].forget()
					_id = self.chem_to_id[smiles]
					yield (_id, smiles, precursors)
				self.pending_results = [res for (i, res) in enumerate(
					self.pending_results) if i not in self.is_ready]
		else:
			def get_pathway_result():
				while not self.pathways_queue.empty():
					yield self.pathways_queue.get(timeout=0.25)
		self.get_pathway_result = get_pathway_result

		# Define how first target is set.
		if self.celery:
			def set_initial_target(_id,smiles):
				self.expand(smiles, 1)
		else:
			def set_initial_target(_id,leaves):
				for leaf in leaves:
					self.active_chemicals[_id].append(leaf)
					self.expand_products(_id, [leaf], self.expansion_branching)			
		self.set_initial_target = set_initial_target

		# Define method to stop working.
		if self.celery:
			def stop():
				if self.pending_results != []:
					import celery.bin.amqp
					from askcos_site.celery import app
					amqp = celery.bin.amqp.amqp(app=app)
					amqp.run('queue.purge', self.private_worker_queue)
				if self.chiral:
					released = tb_c_worker.unreserve_worker_pool.apply_async(queue=self.private_worker_queue, retry=True).get()
				else:
					released = tb_worker.unreserve_worker_pool.apply_async(queue=self.private_worker_queue, retry=True).get()
				self.running = False
		else:
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

	def get_price(self, chem_smi):
		ppg = self.pricer.lookup_smiles(chem_smi, alreadyCanonical=True)
		if ppg:
			return 0.0
		else:
			return None

	def update_tree(self, _id):
		try:
			self.pathway_count += 1
			chemicals = self.pathways[_id]['chemical_nodes']
			reactions = self.pathways[_id]['reaction_nodes']			
			target_smiles = self.pathways[_id]['target']
			smiles_id = self.pathways[_id]['smiles_id']

			# Add in the penalties to the 'purchase price' so they get counted right in Mincost
			for key, C in chemicals.items():
				if C.retro_results == []:
					C.price(self.max_penalty)
					continue
				if key[1] == self.max_depth:
					C.price(self.depth_penalty)
					continue

			# Update costs / successes
			Reset(chemicals,reactions)
			MinCost((target_smiles, 0), self.max_depth, chemicals, reactions)
			target_cost = self.pathways[_id]['chemical_nodes'][(target_smiles,0)].cost
			
			buyable = True
			for chem_key in chemicals:
				if len(chemicals[chem_key].incoming_reactions) == 0:
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
			
			if target_cost == float('inf'):
				for key, C in chemicals.items():
					print key, C.purchase_price, C.cost, C.retro_results
			#print " ------------------------------------------------- "
			
			# Save details for chemicals ... 
			self.save_pathway(self.pathways[_id],target_cost,smiles_id,[c_branching,r_branching],buyable)
		except:
			print "Error in update_tree:", traceback.format_exc()

	def save_pathway(self, pathway, target_cost, target_id, branching, buyable):
		#if self.fileName:
			#with open("train/pathways/" + self.fileName + ".pkl", "a+b") as fid:
			#	pickle.dump(pathway, fid, pickle.HIGHEST_PROTOCOL)
			#with open(self.fileName, "a+") as fid:
			#	fid.write("{} {} {}\n".format(target_id,target_cost,int(buyable)))	

		c_branching, r_branching = branching 
		c_branching = [str(c_branching[k]) for k in range(1,self.max_depth+1)]
		r_branching = [str(r_branching[k]) for k in range(1,self.max_depth+1)]
		branching = c_branching + r_branching
		branching = " ".join(branching)
		print_out = "{} {} {} {}\n".format(target_id,target_cost,int(buyable),branching)

		with open(self.fileName, "a+") as fid:
			fid.write(print_out)	

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
										#print "... put pathway (3) into pathways queue ... "
					
					if finished:
						if all([(len(self.active_chemicals[_id]) == 0) for _id in range(self.nproc)]):
							break
						continue

					for _id in self.get_pathway_result():
						try:
							pair = self.smiles_generator.next()
							smiles_id, smiles = pair
						except StopIteration:
							print "We are finished!"
							finished = True
							break
						leaves = [(smiles, 0)]
						pathway = {'chemicals': set(),
								   		   'chemical_nodes': {},
								   		   'reaction_nodes': {},
								   		   'target': smiles,
								   		   'smiles_id': smiles_id
								   		  }
						self.pathways[_id] = pathway 
						self.pathway_status[_id] = [0, True, self.total_applied_templates]
						self.set_initial_target(_id,leaves)
						elapsed_time = time.time() - start_time

				except Exception as E:
					print "... unspecified ERROR:", traceback.format_exc() 
					elapsed_time = time.time() - start_time

			self.stop()
			print "... exited prematurely."

		except:
			print "Error in coordinate:", traceback.format_exc()
			sys.exit(1)

	def work(self, i):

		use_mincost = False 
		prioritizers = (self.precursor_prioritization, self.template_prioritization)

		if self.precursor_prioritization == gc.mincost:
			print "Loading model weights train/fit/{}/".format(self.policy_iteration)
			from makeit.prioritization.precursors.mincost import MinCostPrecursorPrioritizer
			model = MinCostPrecursorPrioritizer()
			model.load_model(datapath='train/fit/{}/'.format(self.policy_iteration))
			prioritizers = (gc.relevance_precursor, self.template_prioritization)
			use_mincost = True 

		while True:
			# If done, stop
			if self.done.value:
				print 'Worker {} saw done signal, terminating'.format(i)
				#MyLogger.print_and_log(
				#	'Worker {} saw done signal, terminating'.format(i), treebuilder_loc)
				break
			
			# If paused, wait and check again
			if self.paused.value:
				time.sleep(1)
				continue
			
			# Grab something off the queue
			try:
				self.idle[i] = False
				(jj, smiles, depth, branching) = self.expansion_queues[i].get(timeout=0.25)  # short timeout				
				#prioritizers = (self.precursor_prioritization, self.template_prioritization)
				outcomes = self.retroTransformer.get_outcomes(
															smiles, 
															self.mincount, 
															prioritizers,
															depth = depth,
				                                            template_count = self.template_count, 
				                                            mode = self.precursor_score_mode,
				                                            max_cum_prob = self.max_cum_template_prob
				                                            )				
				if use_mincost:
					for precursor in outcomes.precursors:
						precursor.retroscore = 1.0 + sum([abs(model.get_score_from_smiles(smile,depth+1)) for smile in precursor.smiles_list])
						#print smiles, precursor.retroscore, precursor.smiles_list

				reaction_precursors = outcomes.return_top(n=self.rollout_branching) 

				# Epsilon-greedy:
				if (random.random() < self.epsilon) and len(reaction_precursors) > 0:
					reaction_precursors = [random.choice(reaction_precursors)]
				self.results_queues[jj].put((jj, smiles, depth, reaction_precursors))			
			
			except VanillaQueue.Empty:
				pass

			except Exception as e:
				print traceback.format_exc() 
			
			time.sleep(0.01)
			self.idle[i] = True

	def add_reactants(self, _id, chem_smi, depth, precursors):
		try:
			self.pathways[_id]['chemical_nodes'][(chem_smi,depth)].processed = True
			# If no templates applied, do not go further, chemical not makeable.
			if not precursors:
				self.pathways[_id]['chemical_nodes'][chem_smi,depth].retro_results = []
				return 'unexpandable'
				#return False

			scores_list = []
			for result in precursors:
				reactants = result['smiles_split']
				retroscore = result['score']
				template_action = result['tforms']
				template_probability = result['template_score']

				# Reject cyclic templates as 'illegal moves'.
				cyclic_template = False
				for q,smi in enumerate(reactants):		
					if smi in self.pathways[_id]['chemicals']: 
						reactant_smile_key = sorted([(rchem_smi,rdepth) for (rchem_smi,rdepth) in self.pathways[_id]['chemical_nodes'] if (rchem_smi == smi)], key=lambda x: x[1])[0]
						if not (self.pathways[_id]['chemical_nodes'][reactant_smile_key].purchase_price >= 0):
							last_reactant_cost = self.pathways[_id]['chemical_nodes'][reactant_smile_key].cost
							if (last_reactant_cost >= self.max_penalty) or (last_reactant_cost == -1): 
								cyclic_template = True
								break
				if cyclic_template:
					continue
				scores_list.append([retroscore,reactants,template_probability,template_action])
			
			if not scores_list:
				self.pathways[_id]['chemical_nodes'][chem_smi,depth].retro_results = []
				return 'unexpandable'
			
			results = sorted(scores_list, 
							key=lambda x: (x[0], sum([len(xx) for xx in x[1]])))
			self.pathways[_id]['chemical_nodes'][chem_smi,depth].retro_results = results #precursors

			#for result in results:
			#	print chem_smi, depth, result[0], result[1]


			for p,result in enumerate(results):
				react_cost, reactants, template_prob, template_no = result
				if isinstance(reactants, list):
					rxn_smi = ".".join(reactants)
				else:
					rxn_smi = reactants
				
				if p == 0:
					children = []
					self.pathways[_id]['chemical_nodes'][(chem_smi,depth)].add_incoming_reaction(rxn_smi,(template_no,template_prob)) 
					self.pathways[_id]['chemical_nodes'][(chem_smi,depth)].retro_results = result
					self.pathways[_id]['reaction_nodes'][(rxn_smi,depth+1)] = Reaction(rxn_smi,depth+1)
					self.pathways[_id]['reaction_nodes'][(rxn_smi,depth+1)].add_outgoing_chemical(chem_smi,(template_no,template_prob))

					for q,smi in enumerate(reactants):
						if (smi,depth+1) in children: continue
						children.append((smi,depth+1))
						self.pathways[_id]['reaction_nodes'][(rxn_smi,depth+1)].add_incoming_chemical(smi)
						if (smi,depth+1) not in self.pathways[_id]['chemical_nodes']:
							self.pathways[_id]['chemical_nodes'][smi,depth+1] = Chemical(smi,depth+1)
							ppg = self.get_price(smi)
							self.pathways[_id]['chemical_nodes'][smi,depth+1].price(ppg)
							if (ppg >= 0.0) or ((depth + 1) == self.max_depth) or (not self.pathway_status[_id][1]):
								self.pathways[_id]['chemical_nodes'][smi,depth+1].processed = True
							if ((depth + 1) == self.max_depth) and (not ppg >= 0.0):
								self.pathway_status[_id][1] = False
				break

			#################################
			if children and (depth < self.max_depth) and self.pathway_status[_id][1]: #not cyclic_template
				return children
			else:
				#print "Warning (ii): Nothing left to expand.", cyclic_template
				for smi in reactants:
					try:
						if (not self.pathways[_id]['chemical_nodes'][smi,depth+1].processed):
							self.pathways[_id]['chemical_nodes'][smi,depth+1].processed = True
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
			synthetic_expansion_candidates = 0
			for (chem_smi, depth) in children:		
				if depth >= self.max_depth:
					self.active_chemicals[_id].remove((chem_smi,depth))
					self.pathway_status[_id][1] = False
					continue
				
				ppg = self.get_price(chem_smi) #self.Chemicals[chem_smi,depth].purchase_price
				if chem_smi in self.pathways[_id]['chemicals']:		
					if not (ppg >= 0.0):
						self.active_chemicals[_id].remove((chem_smi,depth))
						self.pathway_status[_id][1] = False
						continue
				
				self.pathways[_id]['chemicals'].add(chem_smi)
				if (chem_smi,depth) not in self.pathways[_id]['chemical_nodes']:
					self.pathways[_id]['chemical_nodes'][chem_smi,depth] = Chemical(chem_smi,depth)
				self.pathways[_id]['chemical_nodes'][chem_smi,depth].price(ppg)

				if ppg >= 0: 
					self.pathways[_id]['chemical_nodes'][(chem_smi,depth)].processed = True
					self.active_chemicals[_id].remove((chem_smi,depth))
					continue

				if not (self.pathway_status[_id][0] < self.pathway_status[_id][2]):
					self.pathways[_id]['chemical_nodes'][(chem_smi,depth)].processed = True
					self.active_chemicals[_id].remove((chem_smi,depth))
					self.pathway_status[_id][1] = False
					continue
				
				synthetic_expansion_candidates += 1
				self.pathway_status[_id][0] += 1
				
				# TO-DO
				# If we have already expanded the node, don't re-do it.
				# Form for results_queue.put(): (jj, smiles, depth, precursors, pathway)
				self.expand(_id,chem_smi,depth,branching)
			
			return synthetic_expansion_candidates

		except Exception as e:
			print "Error in expand_products:", traceback.format_exc()

	def target_generator(self):
		return self.target_generator_func()

	def target_generator_func(self):
		for data in self.target_chemicals:
			yield data

	def build_tree(self):
		start_time = time.time()
		self.running = True
		self.prepare()
		
		self.smiles_generator = self.target_generator()
		for k in range(self.nproc):
			try:
				pair = self.smiles_generator.next()
				smiles_id, smiles = pair
				#self.epsilon = epsilon
			except StopIteration:
				print "(a) We are finished!"
				break
			leaves = [(smiles, 0)]
			pathway = {'chemicals': set(),
			   		   'chemical_nodes': {},
			   		   'reaction_nodes': {},
			   		   'target': smiles,
			   		   'smiles_id': smiles_id
					  }
			self.pathways[k] = pathway 
			self.pathway_status[k] = [0, True, self.total_applied_templates]
			self.set_initial_target(k,leaves)

		# Coordinate workers.
		self.coordinate()
		
		'''
		# Save CRN
		mincost, num_pathways = self.save_crn()

		# Save states for training value network 
		training_states_save = "states/replica_{}.pickle".format(self.replica)
		value_network_training_states(self.smiles_id, 
										self.Chemicals, 
										self.Reactions, 
										FP_rad = 3, 
										FPS_size = 16384, 
										fileName = training_states_save)
		'''
		print "Finished working."

	def reset(self):
		if self.celery:
			# general parameters in celery format
			pass
		else:
			self.manager = Manager()
			self.done = self.manager.Value('i', 0)
			self.paused = self.manager.Value('i', 0)
			self.idle = self.manager.list()
			self.results_queue = Queue()
			self.workers = []
			self.coordinator = None
			self.running = False
			
			## Queues 
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
							fileName = None, 
							max_depth=10, 
							expansion_time = 300,
							expansion_branching = 1, 
							rollout_branching = 1,
							total_applied_templates = 1000,
							noise_std_percentage = None,
							template_prioritization=gc.relevance, 
				 			precursor_prioritization=gc.heuristic, 
				 			policy_iteration = None,
				 			nproc=8, mincount=25, chiral=True, epsilon = 0.0, 
                          	template_count=50, precursor_score_mode=gc.max, 
                          	max_cum_template_prob = 0.995):
		self.target_chemicals = target_chemicals
		self.replica = replica 
		self.fileName = fileName 
		self.mincount = mincount
		self.max_depth = max_depth
		self.expansion_branching = expansion_branching
		self.expansion_time = expansion_time
		self.rollout_branching = rollout_branching 
		self.total_applied_templates = total_applied_templates
		self.template_prioritization = template_prioritization
		self.precursor_prioritization = precursor_prioritization
		self.precursor_score_mode = precursor_score_mode
		self.nproc = nproc
		self.template_count = template_count
		self.max_cum_template_prob = max_cum_template_prob
		self.epsilon = epsilon
		if self.precursor_prioritization == 'random':
			self.epsilon = 1.0
		self.noise_std_percentage = noise_std_percentage
		self.policy_iteration = policy_iteration

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
		self.pathways = [0 for i in range(self.nproc)]
		self.pathways_queue = Queue()
		self.pathway_status = [[0,True,self.total_applied_templates] for i in range(self.nproc)]
		self.sampled_pathways = []
		self.pathway_count = 0 
		self.successful_pathway_count = 0

		if not self.celery:
			for i in range(nproc):
				self.idle.append(True)
			if self.nproc != 1:
				self.expansion_queues = [Queue() for i in range(self.nproc)]
				self.results_queues   = [Queue() for i in range(self.nproc)]
			else:
				self.expansion_queues = [Queue()]
				self.results_queues   = [Queue()]
		self.active_chemicals = [[] for x in range(nproc)]

		#print "Starting search for id:", smiles_id, "smiles:", smiles
		return self.build_tree()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--policy', type = str, default = 'heuristic',
						help = 'The policy to use for selecting reactant precursors')
	parser.add_argument('--nodes', type = int, default = 50,
						help = 'How many CPU nodes to use')
	parser.add_argument('--epsilon', type = float, default = 0.2,
		                help = 'Value to use in 1-e greedy searching.')
	parser.add_argument('--iteration', type = int, default = 1,
		                help = 'Which policy iteration weights to use')
	parser.add_argument('--replica', type = int, default = 0,
		                help = 'Replica ID.')
	parser.add_argument('--copies', type = int, default = 50,
		                help = 'Number of times to run each chemical sample. For use with 1-e searching.')
	args = parser.parse_args()

	start_time = time.time()
	simulation_time = 120.0 * 3600.0 - 60.0

	policy	           = str(args.policy)
	N_nodes 	       = int(args.nodes)
	epsilon            = float(args.epsilon)
	iteration          = int(args.iteration)
	replica 		   = int(args.replica)
	copies  		   = int(args.copies)

	if policy == 'heuristic':
		policy = gc.heuristic
	elif policy == 'relevance_precursor':
		policy = gc.relevance_precursor
	elif policy == 'mincost':
		policy = gc.mincost
	else:
		print "You need to pick a valid precursor priortizer ... exiting"
		sys.exit()

	# Load training samples
	smiles_to_index = "states/training_smiles.txt"
	smiles_ID_mapping = {}
	with open(smiles_to_index, "r") as fid:
		for line in fid.readlines():
                        try:
                                l1, l2 = line.strip("\n").split(" ")
                        except:
                                l1, l2, l3, l4 = line.strip("\n").split(" ")
			smiles_ID_mapping[l1] = l2
	training_samples = sorted(smiles_ID_mapping.items(), key = lambda x: int(x[0]))
	
	if copies > 1:
		new_list = []
		for sample in training_samples:
			for k in range(copies):
				new_list.append(sample)
		training_samples = new_list


        #training_samples = [['294', '[H]C1(CCC(C)(O1)C1([H])CCC2([H])OC([H])(CCC2(C)O1)C1(C)CCC(Br)C(C)(C)O1)C1(C)CCC(O1)C(C)(C)O']]

	N_samples = len(training_samples) 
	batch_size = int(N_samples / N_nodes) + 1 
	training_samples = training_samples[replica * batch_size: (replica+1) * batch_size]
        print "Total samples:", N_samples, "samples per sim:", len(training_samples)

	if policy == gc.mincost:
		logger_file = "ability/logs/AAA_outcomes_{}_{}_{}_{}.txt.training".format(policy,iteration,epsilon,replica)
	else:
		logger_file = "ability/logs/outcomes_{}_{}_{}.txt.buyable".format(policy,epsilon,replica)

	# Load tree builder 
	NCPUS = psutil.cpu_count()
	print "There are {} processes available ... ".format(NCPUS)
	Tree = MCTS(nproc=NCPUS)
	
	result = Tree.get_buyable_paths(training_samples, 
									replica = replica, 
									nproc = NCPUS,
									fileName = logger_file,
									expansion_time = simulation_time,
									precursor_prioritization = policy,
									rollout_branching = 50,
									epsilon = epsilon,
									policy_iteration = iteration
									)
