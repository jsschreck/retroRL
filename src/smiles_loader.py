
class smilesLoader:

    def __init__(self, smilesFile, verbose = True):
        self.smilesFile = smilesFile
        self.verbose = verbose

    def smiles_list(self):
        training_samples = []
        with open(self.smilesFile, "r") as fid:
            for line in fid.readlines():
                l1, l2 = line.strip("\n").split(" ")[:2]
                training_samples.append([l1,l2])
        return training_samples

    def replica_load(self, N_nodes, replica):
        training_samples = self.smiles_list()
        N_samples = len(training_samples)
        if self.verbose:
            print "Number of targets:", N_samples
        batch_size = int(N_samples / N_nodes) + 1
        training_samples = training_samples[
            replica * batch_size: (_replica+1) * batch_size
            ]
        N_samples = len(training_samples)
        if self.verbose:
            print "Number of targets on this replica:", N_samples
        return training_samples
