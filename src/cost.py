import os, sys 

def score_max_depth(smiles=""):
	return 10.

def score_no_templates(smiles=""):
    return 100.

def RSF(smiles_list=[]):
	return 1.

def Reset(Chemicals,Reactions):
	"""
	Reset cost info
	"""
	for key in Chemicals.keys(): Chemicals[key].reset()
	for key in Reactions.keys(): Reactions[key].reset()

def BuyablePathwayCount(chem_key, max_depth, Chemicals, Reactions):
    '''
    Computes the total number of buyable routes in a graph G=(C,R).
    '''
    Chemical = Chemicals[chem_key] 
    current_depth = chem_key[1]
    if Chemical.counter < 0:
        if len(Chemical.incoming_reactions) == 0 and Chemical.purchase_price == 0:
            Chemical.counter = 1
        elif len(Chemical.incoming_reactions) == 0 and (Chemical.purchase_price == None or Chemical.purchase_price == -1):
            Chemical.counter = 0
            return Chemical.counter
        else:
            Chemical.counter = 0 
        if current_depth < max_depth:
            for reac in Chemical.incoming_reactions:
                rxnsmi, d, _id = reac 
                r = Reactions[(rxnsmi,d)]
                if r.mark == 0:
                    if r.counter < 0:
                        r.counter = 1
                        r.mark    = 1
                        for reactant in r.incoming_chemicals:
                            r.counter = r.counter * BuyablePathwayCount(tuple(reactant),
                                max_depth,Chemicals,Reactions)
                        r.mark = 0 
                    Chemical.counter += r.counter
    return Chemical.counter

def MinCost(chem_key, max_depth, Chemicals, Reactions):
    '''
    Computes the cheapest pathway
    ''' 
    Chemical = Chemicals[chem_key] 
    current_depth = chem_key[1]
    if Chemical.cost < 0:        
        if Chemical.purchase_price == None or Chemical.purchase_price == -1:
            Chemical.cost = float('inf')
            #if (current_depth == max_depth):
            #    Chemical.cost = score_max_depth(Chemical.smiles) #(10**7)*float(len(Chemical.smiles))
            #else:
            #    Chemical.cost = score_no_templates(Chemical.smiles)
        else:
            Chemical.cost = Chemical.purchase_price
        
        if current_depth < max_depth:
            for reac in Chemical.incoming_reactions:
                rxnsmi, d, _id = reac 
                r = Reactions[(rxnsmi,d)]
                if r.mark == 0:
                    if r.cost < 0:
                        product_smiles  = [Chemical.smiles]
                        reactant_smiles = r.smiles.split(".")
                        r.cost = RSF()
                        r.mark = 1
                        for reactant in r.incoming_chemicals:
                            r.cost = r.cost + MinCost(tuple(reactant),max_depth,
                                Chemicals,Reactions)
                        r.mark = 0 
                    if r.cost < Chemical.cost:
                        Chemical.cost = r.cost
    return Chemical.cost

if __name__ == '__main__':
    import cPickle as pickle 
    with open("test.pkl", "rb") as fid:
        chemicals, reactions = pickle.load(fid)

    smi = ('[H]C1(CCC(C)(O1)C1([H])CCC2([H])OC([H])(CCC2(C)O1)C1(C)CCC(Br)C(C)(C)O1)C1(C)CCC(O1)C(C)(C)O', 0)
    #chemicals = pathway['chemical_nodes']
    #reactions = pathway['reaction_nodes']
    Reset(chemicals,reactions)
    MinCost(smi,10,chemicals,reactions)
    print "===================="
    for c,C in chemicals.items():
        print c, C.purchase_price, C.cost 
    print "--------------------"
    for r,R in reactions.items():
        print r, R.cost

