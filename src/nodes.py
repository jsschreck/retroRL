class Chemical(object):
    """Represents a chemical compound."""

    def __init__(self, chem_smi, depth):
        # Initialize lists for incoming and outgoing reactions.
        self.smiles = chem_smi
        self.depth  = depth 
        self.retro_results = None 
        self.transform_ids = []
        self.incoming_reactions = []
        self.outgoing_reactions = []
        self.purchase_price = -1
        self.processed = False
        self.visit_count = 0
        self.rewards = []

        # Counter param used for the DFS search. 
        self.cost    = -1 
        self.counter = -1
        self.makeable = False 

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.smiles)

    def __str__(self):
        return "%s" % self.smiles

    def add_incoming_reaction(self, incoming_smiles, transform_id):
        if [incoming_smiles,self.depth+1,transform_id] not in self.incoming_reactions:
            self.incoming_reactions.append([incoming_smiles,self.depth+1,transform_id])

    def add_outgoing_reaction(self, outgoing_smiles):
        if [outgoing_smiles,self.depth] not in self.outgoing_reactions:
            self.outgoing_reactions.append([outgoing_smiles,self.depth])

    def price(self, value):
        try:
            ppg = float(value)
            self.purchase_price = ppg
        except:
            pass

    def reset(self):
        self.cost = -1
        self.counter = -1
        self.makeable = False

class Reaction(object):
    """Represents a reaction."""

    def __init__(self, smiles, depth):
        """Initialize entry."""
        self.smiles = smiles.strip()
        self.depth  = depth 
        self.incoming_chemicals = []
        self.outgoing_chemicals = []
        self.purchase_price = float("inf")
        self.visit_count = 0
        self.cost_estimate = float("inf")
        self.successes = []
        self.rewards = []

        # Counter and marking params used for the DFS search. 
        self.mark = 0
        self.cost = -1 
        self.viable = False 
        self.counter = -1 

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.smiles)

    def __str__(self):
        return "%s" % self.smiles

    def add_incoming_chemical(self, incoming_smiles):
        if [incoming_smiles,self.depth] not in self.incoming_chemicals:
            self.incoming_chemicals.append([incoming_smiles,self.depth])

    def add_outgoing_chemical(self, outgoing_smiles, transform_id):
        if [outgoing_smiles,self.depth-1,transform_id] not in self.outgoing_chemicals:
            self.outgoing_chemicals.append([outgoing_smiles,self.depth-1,transform_id])

    def reaction_price(self, value):
        if not isinstance(value,float):
            self.purchase_price = float("inf")
        else:
            self.purchase_price = value

    def reset(self):
        self.mark = 0
        self.cost = -1 
        self.viable = False 
        self.counter = -1 

