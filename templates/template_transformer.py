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
