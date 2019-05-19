
class smilesLoader:

    def __init__(self, smilesFile, verbose=True):
        self.smilesFile = smilesFile
        self.verbose = verbose

    def smiles_list(self):
        training_samples = []
        with open(self.smilesFile, "r") as fid:
            for line in fid.readlines():
                l1, l2 = line.strip("\n").split(" ")[:2]
                training_samples.append([l1, l2])
        return training_samples

    def load(self, N_nodes, replica):
        training_samples = self.smiles_list()
        N_samples = len(training_samples)
        if self.verbose:
            print "Number of targets:", N_samples
        batch_size = int(N_samples / N_nodes) + 1
        training_samples = training_samples[
            replica * batch_size: (replica+1) * batch_size
        ]
        N_samples = len(training_samples)
        if self.verbose:
            print "Number of targets on this replica:", N_samples
        return training_samples


def ResetCostEstimate(Chemicals, Reactions, reset_only_estimate=False):
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
