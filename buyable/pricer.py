from collections import defaultdict
import rdkit.Chem as Chem
import cPickle as pickle
import os, sys

class Pricer:
    '''
    ---> This is Connor Coley's Pricer class <---

    The Pricer class is used to look up the ppg of chemicals if they
    are buyable.
    '''
    def load_from_file(self, file_path = 'buyable/buyable'):
        '''
        Load the data for the pricer from a locally stored file instead of from an online database.
        '''
        if not os.path.isfile(file_path):
            filepath = 'buyable'
        
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                self.prices = pickle.load(file)
                self.prices_flat = pickle.load(file)
                self.prices_by_xrn = pickle.load(file)
        else:
            print("ALERT: You need to supply a pricer details.")
            print("Current path:", sys.argv[0])
            sys.exit(1)

    def lookup_smiles(self, smiles, alreadyCanonical=False, isomericSmiles=True):
        '''
        Looks up a price by SMILES. Tries it as-entered and then 
        re-canonicalizes it in RDKit unl ess the user specifies that
        the string is definitely already canonical.
        '''
        ppg = self.prices_flat[smiles]
        if ppg:
            return ppg

        ppg = self.prices[smiles]
        if ppg:
            return ppg

        if not alreadyCanonical:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0.
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)

            ppg = self.prices_flat[smiles]
            if ppg:
                return ppg

            ppg = self.prices[smiles]
            if ppg:
                return ppg

        return ppg
