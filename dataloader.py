from torch.utils.data import Dataset
import numpy as np

class bitermsDataset(Dataset):
    def __init__(self, biterms, num_input, mini_doc=10, data=None):
        self.data = data
        self.biterms = biterms
        self.num_input = num_input
        self.rng = np.random.RandomState(10)
        self.length = len(self.biterms)
        self.mini_doc = mini_doc

    def __len__(self):
        return len(self.biterms)

    def _make_graph(self, corpus):
        merge = {}
        for d in corpus:
            merge = {**merge, **d}
        temp = np.zeros((self.num_input, self.num_input), dtype='float32')
        for k, v in merge.items():
            if len(k) < 2:
                a = b = list(k)[0]
            else:
                a, b = k
            temp[a, b] = v
            temp[b, a] = v
        return temp

    def __getitem__(self, idx):
        ixs = self.rng.randint(self.length, size=self.mini_doc)

        target = self._make_graph(self.biterms[ixs])
        return target


