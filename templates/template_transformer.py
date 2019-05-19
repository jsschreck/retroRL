from __future__ import print_function

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from retroRL.prioritizers.heuristic import HeuristicPrecursorPrioritizer
from retroRL.prioritizers.relevance import RelevanceTemplatePrioritizer
from retroRL.prioritizers.cost import CostPrecursorPrioritizer
from retroRL.prioritizers.default import DefaultPrioritizer

from rdchiral.initialization import rdchiralReaction, rdchiralReactants

import cPickle as pickle 
import os, sys

retro_transformer_loc = 'retro_transformer'

class TemplateTransformer(object):
    '''
    The Transformer class defines an object which can be used to perform
    one-step retrosyntheses for a given molecule.
    '''

    def __init__(self):
        self.id_to_index = {} # Dictionary to keep track of ID -> index in self.templates

    def get_precursor_prioritizers(self, precursor_prioritizer):
        if not precursor_prioritizer:
            print(
                'Cannot run the Transformer without a precursor prioritization method. Exiting...', transformer_loc)
        if precursor_prioritizer in self.precursor_prioritizers:
            precursor = self.precursor_prioritizers[precursor_prioritizer]
        else:
            if precursor_prioritizer == 'Heuristic':
                precursor = HeuristicPrecursorPrioritizer()
            elif precursor_prioritizer == 'Cost':
                precursor = CostPrecursorPrioritizer()
            else:
                precursor = DefaultPrioritizer()
                print(
                    'Prioritization method not recognized. Using natural prioritization.')

            precursor.load_model()
            self.precursor_prioritizers[precursor_prioritizer] = precursor

        self.precursor_prioritizer = precursor

    def get_template_prioritizers(self, template_prioritizer):
        if not template_prioritizer:
            print(
                'Cannot run the Transformer without a template prioritization method. Exiting...')
        if template_prioritizer in self.template_prioritizers:
            template = self.template_prioritizers[template_prioritizer]
        else:
            if template_prioritizer == 'Relevance':
                template = RelevanceTemplatePrioritizer()
            else:
                print('Prioritization method not recognized.')
                sys.exit()
                
            template.load_model()
            self.template_prioritizers[template_prioritizer] = template

        self.template_prioritizer = template

    def load_from_file(self, retro, file_path, chiral=False, rxns=True, refs=False, efgs=False, rxn_ex=False):
        '''
        Read the template database from a previously saved file, of which the path is specified in the general
        configuration
        retro: whether in the retrosynthetic direction
        file_path: .pickle file to read dumped templates from 
        chiral: whether to handle chirality properly (only for retro for now)
        rxns : whether or not to actually load the reaction objects (or just the info)
        '''
        
        print('Loading templates from {}'.format(file_path))

        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                if retro and chiral and rxns: # cannot pickle rdchiralReactions, so need to reload from SMARTS
                    pickle_templates = pickle.load(file)
                    self.templates = []
                    for template in pickle_templates:
                        try:
                            template['rxn'] = rdchiralReaction(
                                str('(' + template['reaction_smarts'].replace('>>', ')>>(') + ')'))
                        except Exception as e:
                            template['rxn'] = None
                        self.templates.append(template)
                else:
                    self.templates = pickle.load(file)
        else:
            print("No file to read data from.")
            raise IOError('File not found to load template_transformer from!')

        # Clear out unnecessary info
        if not refs:
            [self.templates[i].pop('references', None) for i in range(len(self.templates))]
        elif 'references' not in self.templates[0]:
            raise IOError('Save file does not contain references (which were requested!)')

        if not efgs:
            [self.templates[i].pop('efgs', None) for i in range(len(self.templates))]
        elif 'efgs' not in self.templates[0]:
            raise IOError('Save file does not contain efg info (which was requested!)')

        if not rxn_ex:
            [self.templates[i].pop('rxn_example', None) for i in range(len(self.templates))]
        elif 'rxn_example' not in self.templates[0]:
            raise IOError('Save file does not contain a reaction example (which was requested!)')


        self.num_templates = len(self.templates)
        print('Loaded templates. Using {} templates'.format(self.num_templates))

    def get_prioritizers(self, *args, **kwargs):
        '''
        Get the prioritization methods for the transformer (templates and/or precursors)
        '''
        raise NotImplementedError

    def load(self, *args, **kwargs):
        '''
        Load and initialize templates
        '''
        raise NotImplementedError

    def reorder(self):
        '''Reorder self.templates in descending popularity. Also builds id_to_index table'''
        self.num_templates = len(self.templates)
        self.templates = sorted(self.templates, key=lambda z: z[
                                'count'], reverse=True)
        self.id_to_index = {template['_id']: i for i,
                            template in enumerate(self.templates)}
        return

    def lookup_id(self, template_id):
        '''
        Find the reaction smarts for this template_id
        '''

        if not self.id_to_index:  # need to build
            self.id_to_index = {template['_id']: i for (
                i, template) in enumerate(self.templates)}
        return self.templates[self.id_to_index[template_id]]

    def load_from_database(self, retro, chiral=False, refs=False, rxns=True, efgs=False, rxn_ex=False):
        # Save collection TEMPLATE_DB
        self.load_databases(retro, chiral=chiral)
        self.chiral = chiral
        if self.mincount and 'count' in self.TEMPLATE_DB.find_one():
            if retro:
                filter_dict = {'count': {'$gte': min(
                    self.mincount, self.mincount_chiral)}}
            else:
                filter_dict = {'count': {'$gte': self.mincount}}
        else:
            filter_dict = {}

        # Look for all templates in collection
        to_retrieve = ['_id', 'reaction_smarts',
                       'necessary_reagent', 'count', 'intra_only', 'dimer_only']
        if refs:
            to_retrieve.append('references')
        if efgs:
            to_retrieve.append('efgs')
        if rxn_ex:
            to_retrieve.append('rxn_example')
        for document in self.TEMPLATE_DB.find(filter_dict, to_retrieve):
            # Skip if no reaction SMARTS
            if 'reaction_smarts' not in document:
                continue
            reaction_smarts = str(document['reaction_smarts'])
            if not reaction_smarts:
                continue

            if retro:
                # different thresholds for chiral and non chiral reactions
                chiral_rxn = False
                for c in reaction_smarts:
                    if c in ('@', '/', '\\'):
                        chiral_rxn = True
                        break

                if chiral_rxn and document['count'] < self.mincount_chiral:
                    continue
                if not chiral_rxn and document['count'] < self.mincount:
                    continue

            # Define dictionary
            template = {
                'name':                 document['name'] if 'name' in document else '',
                'reaction_smarts':      reaction_smarts,
                'incompatible_groups':  document['incompatible_groups'] if 'incompatible_groups' in document else [],
                'reference':            document['reference'] if 'reference' in document else '',
                'references':           document['references'] if 'references' in document else [],
                'rxn_example':          document['rxn_example'] if 'rxn_example' in document else '',
                'explicit_H':           document['explicit_H'] if 'explicit_H' in document else False,
                '_id':                  document['_id'] if '_id' in document else -1,
                'product_smiles':       document['product_smiles'] if 'product_smiles' in document else [],
                'necessary_reagent':    document['necessary_reagent'] if 'necessary_reagent' in document else '',
                'efgs':                 document['efgs'] if 'efgs' in document else None,
                'intra_only':           document['intra_only'] if 'intra_only' in document else False,
                'dimer_only':           document['dimer_only'] if 'dimer_only' in document else False,
            }
            if retro:
                template['chiral'] = chiral_rxn

            # Frequency/popularity score
            if 'count' in document:
                template['count'] = document['count']
            elif 'popularity' in document:
                template['count'] = document['popularity']
            else:
                template['count'] = 1

            # Define reaction in RDKit and validate
            if rxns:
                try:
                    # Force reactants and products to be one pseudo-molecule (bookkeeping)
                    reaction_smarts_one = '(' + reaction_smarts.replace('>>', ')>>(') + ')'

                    if retro:
                        if chiral:
                            rxn = rdchiralReaction(str(reaction_smarts_one))
                            template['rxn'] = rxn
                        else:
                            rxn = AllChem.ReactionFromSmarts(
                                str(reaction_smarts_one))
                            if rxn.Validate()[1] == 0:
                                template['rxn'] = rxn
                            else:
                                template['rxn'] = None
                    else:
                        rxn_f = AllChem.ReactionFromSmarts(reaction_smarts_one)
                        if rxn_f.Validate()[1] == 0:
                            template['rxn_f'] = rxn_f
                        else:
                            template['rxn_f'] = None

                except Exception as e:
                    template['rxn'] = None
                    template['rxn_f'] = None

            # Add to list
            self.templates.append(template)

        self.reorder()

    def get_outcomes(self, *args, **kwargs):
        '''
        Performs a one-step transformation given a SMILES string of a
        target molecule by applying each transformation template
        sequentially.
        '''
        raise NotImplementedError

    def apply_one_template(self, *args, **kwargs):
        '''
        Takes a mol object and applies a single template, returning
        a list of precursors or outcomes, depending on whether retro or 
        synthetic templates are used
        '''
        raise NotImplementedError
