from retroRL.prioritizers.prioritizer import Prioritizer
from retroRL.buyable.pricer import Pricer

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np

import math
import sys
import random
import os
import time
import os
import cPickle as pickle
from numpy import inf

class CostPrecursorPrioritizer(Prioritizer):
    '''
    This is a standalone, importable Cost model. Uses loaded keras model.
    '''

    def __init__(self, score_scale=100.0, max_depth = 10):
        self.vars = []
        self.FP_rad = 3
        self.max_depth = max_depth
        self.score_scale = score_scale
        
        self._restored = False
        self.pricer = None
        self._loaded = False
 
    def load_model(self, FP_len=16384, input_layer=1024, hidden_layer=300, datapath=""):
        
        from keras.models import Sequential, model_from_json
        from keras.layers import Dense, Lambda, Activation
        from keras import backend as K
        
        self.FP_len = FP_len

        modelpath = "../prioritizers/cost/model.json"
        weightpath = "../prioritizers/cost/weights.h5"
        if not os.path.isfile(modelpath):
            modelpath = "prioritizers/cost/model.json"
            weightpath = "prioritizers/cost/weights.h5"
        if not os.path.isfile(modelpath):
            print "Cannot load model. Check path"
            sys.exit(1)
    
        model = model_from_json(open(modelpath).read())
        model.load_weights(weightpath)
        self.model = model 
        
        def mol_to_fp(mol, depth):
            if mol is None:
                _fps = np.zeros((self.FP_len,), dtype=np.float32)
            else:
                _fps = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.FP_rad, 
                                                                      nBits=self.FP_len,
                                                                      useChirality=True, 
                                                                      ), 
                                dtype=np.float32)
            return np.hstack([_fps, self.max_depth - int(depth)])
        self.mol_to_fp = mol_to_fp

        self.pricer = Pricer()
        if sys.argv[0] == 'cost.py':
            self.pricer.load_from_file('../buyable/buyable')
        else:
            self.pricer.load_from_file('buyable/buyable')
        self._restored = True
        self._loaded = True

    def smi_to_fp(self, smi, depth):
        if not smi:
            return np.hstack([np.zeros((self.FP_len + 1,), dtype=np.float32), self.max_depth - int(depth)])
        return self.mol_to_fp(Chem.MolFromSmiles(smi), depth)

    def get_price(self, smi):
        ppg = self.pricer.lookup_smiles(smi, alreadyCanonical=True)
        if ppg:
            return 0.0
        else:
            return None 

    def get_priority(self, retroProduct, depth = None, **kwargs):
        if not self._loaded:
            self.load_model()

        if not isinstance(retroProduct, str):
            scores = []
            depth = int(retroProduct.depth)
            for smiles in retroProduct.smiles_list:
                scores.append(self.get_score_from_smiles(smiles,depth))
            return -sum(scores)
        else:
            if type(depth) != int:
                print "You need to supply depth argument. Exiting..."
                sys.exit()
            depth = int(depth)
            return -self.get_score_from_smiles(retroProduct,depth)
        if not retroProduct:
            return -inf

    def get_score_from_smiles(self, smiles, depth, pricer = True):
        # Check buyable
        if pricer:
            ppg = self.pricer.lookup_smiles(smiles, alreadyCanonical=True)
            if ppg:
                return 0.0 #ppg / 100.
        
        fp = np.array((self.smi_to_fp(smiles, depth)), dtype=np.float32)
        if sum(fp) == 0:
            cur_score = 0.
        else:
            fp = fp.reshape(-1, fp.shape[0])
            cur_score = self.model.predict(fp)[0][0]
        return 100.0 * cur_score

    def get_raw_score_from_smiles(self, smiles, depth):    
        fp = np.array((self.smi_to_fp(smiles, depth)), dtype=np.float32)
        if sum(fp) == 0:
            cur_score = 0.
        else:
            fp = fp.reshape(-1, fp.shape[0])
            cur_score = self.model.predict(fp)[0][0]
        return 100.0 * cur_score

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.abspath('../'))
    model = CostPrecursorPrioritizer()
    model.load_model()
    if len(sys.argv) == 1:
        smis = [('COC1C=CC(=CC=1)C1=CC(C)=CC(=N1)C1C=CC(=CC=1)OC', 0), ('CN(CC1C=CC=CC=1)CC1CCCC1=O', 0), ('NC1C=CN(C(=O)N=1)C1OC(COC(=O)CCCCC(=O)NC2C=CC=CC=2)C(O)C1(F)F', 0), ('CC(=O)N1C=C(C=C2N=C(N(N=CC3C=CC=CC=3)C(C)=O)N(C(C)=O)C2=O)C2=CC=CC=C21', 10), ('CCCNc1ccccc1', 10)]
    else:
        smis = sys.argv[1:]
    for (smi,depth) in smis:
        sco = abs(model.get_priority(smi, depth = depth))
        print('{} <--- ({},{})'.format(sco, smi, depth))
