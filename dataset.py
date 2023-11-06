from gensim.models import Word2Vec
from torch.utils import data
from tqdm import tqdm
import numpy as np
from typing import List
import torch
from collections import Counter
from sklearn_extra.cluster import KMedoids


class TrainDataset(data.Dataset):
    def __init__(self, args, w2v):
        super(TrainDataset, self).__init__()
        self.train_data = torch.load(args.train_data)
        self.w2id = {w: w2v.wv.vocab[w].index for w in w2v.wv.vocab}
        self.pad_idx = w2v.wv.vocab.get("<pad>").index
        self.maxlen = args.maxlen
        self.maxlen_seed = args.maxlen_seed
        self.w2v = w2v
        self.maxlen_seed_aspects = args.maxlen_seed_aspects
        self.general_idx = args.general_idx
        self.num_cluster_general = args.num_cluster_general
        self.num_cluster_aspects = args.num_cluster_aspects
        self.all_seed, self.num_clusters, self.num_arr = self.load_file(args.seed_data)

    def load_file(self, file: str):
        seeds = []
        num_clusters = []
        len_dcts = []
        dt = []
        with open(file, "r") as f:
            for line in f:
                if len(line.strip().split()) > 0:
                    dt.append(line.strip())

        # Take seed aspects-general
        dt = dt = [
            " ".join(dt[i].split()[: self.maxlen_seed])
            if i == self.general_idx
            else " ".join(dt[i].split()[: self.maxlen_seed_aspects])
            for i in range(len(dt))
        ]

        # --------END---------------

        tmp = []
        for item in dt:
            tmp.extend(item.split())
        union = []
        l = Counter(tmp)
        for wd, nm in l.items():
            if nm > 1:
                union.append(wd)
        for i in range(len(dt)):
            num_clus = -1
            if i == self.general_idx:
                num_clus = self.num_cluster_general
            else:
                num_clus = self.num_cluster_aspects
            num_max = 0
            for sd in dt[i].split():
                if sd in self.w2v.wv.vocab:
                    num_max += 1
            if num_clus > num_max:
                num_clus = num_max
            s = dt[i].split()
            dct = dict([(i, []) for i in range(num_clus)])
            X = np.asarray([self.w2v[w] for w in s if w in self.w2v.wv.vocab])
            kmedoids = KMedoids(n_clusters=num_clus, random_state=0).fit(X)
            for i in range(len(X)):
                if (s[i] not in union) and (s[i] in self.w2v.wv.vocab):
                    dct[kmedoids.labels_[i]].append(s[i])
            len_dct = 0
            seed = []
            for k, v in dct.items():
                if len(v) > 0:
                    len_dct += 1
                    seed.append(" ".join(v))
            seeds.extend(seed)
            dcts = [len_dct] * len(seed)
            len_dcts.extend(dcts)
            num_clusters.append(len_dct)
        return seeds, num_clusters, len_dcts

    def sent2idx(self, tokens: List[str], is_sent):
        # tokens = tokens.split()
        if is_sent:
            tokens = [
                self.w2id[token.strip()]
                for token in tokens
                if token in self.w2id.keys()
            ]
            tokens += [self.pad_idx] * (self.maxlen - len(tokens))
            tokens = tokens[: self.maxlen]
        else:
            tokens = [
                self.w2id[token.strip()]
                for token in tokens
                if token in self.w2id.keys()
            ]
            ratio = int(self.maxlen_seed / len(tokens)) + 1
            tokens = tokens * ratio
            tokens = tokens[: self.maxlen_seed]
        return tokens

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, id):
        sent = self.sent2idx(self.train_data[id][0], 1)
        self.mask = [i != self.pad_idx for i in sent]
        if np.array(self.mask).max() == 0:
            self.mask = np.concatenate(([1], np.zeros(len(self.mask) - 1)))
        seeds = []
        for item in self.all_seed:
            seeds.append(self.sent2idx(item.split(), 0))

        return (
            torch.LongTensor(sent),
            torch.LongTensor(seeds),
            torch.as_tensor(self.num_clusters),
            torch.as_tensor(self.num_arr),
            torch.as_tensor(np.array(self.mask).astype(int)),
        )  # , torch.LongTensor(label), torch.tensor(reli)


class TestDataset(data.Dataset):
    def __init__(self, args, w2v):
        super(TestDataset, self).__init__()
        self.test_data = torch.load(args.test_data)
        self.maxlen = args.maxlen
        self.maxlen_seed = args.maxlen_seed
        if "<pad>" in w2v.wv.vocab:
            self.pad_idx = w2v.wv.vocab.get("<pad>").index
        else:
            self.pad_idx = w2v.wv.vocab.get("etc").index
        self.w2id = {w: w2v.wv.vocab[w].index for w in w2v.wv.vocab}
        self.general_idx = args.general_idx
        self.num_cluster_general = args.num_cluster_general
        self.num_cluster_aspects = args.num_cluster_aspects
        self.w2v = w2v
        self.maxlen_seed_aspects = args.maxlen_seed_aspects
        self.all_seed, self.num_clusters, self.num_arr = self.load_file(args.seed_data)

    def load_file(self, file: str):
        seeds = []
        num_clusters = []
        len_dcts = []
        dt = []
        with open(file, "r") as f:
            for line in f:
                if len(line.strip().split()) > 0:
                    dt.append(line.strip())

        # Take seed aspects-general
        dt = dt = [
            " ".join(dt[i].split()[: self.maxlen_seed])
            if i == self.general_idx
            else " ".join(dt[i].split()[: self.maxlen_seed_aspects])
            for i in range(len(dt))
        ]
        # --------END---------------

        tmp = []
        for item in dt:
            tmp.extend(item.split())
        union = []
        l = Counter(tmp)
        for wd, nm in l.items():
            if nm > 1:
                union.append(wd)
        for i in range(len(dt)):
            num_clus = -1
            if i == self.general_idx:
                num_clus = self.num_cluster_general
                num_max = self.num_cluster_general
            else:
                num_clus = self.num_cluster_aspects
            num_max = 0
            for sd in dt[i].split():
                if sd in self.w2v.wv.vocab:
                    num_max += 1
            if num_clus > num_max:
                num_clus = num_max
            s = dt[i].split()
            dct = dict([(i, []) for i in range(num_clus)])
            X = np.asarray([self.w2v[w] for w in s if w in self.w2v.wv.vocab])
            kmedoids = KMedoids(n_clusters=num_clus, random_state=0).fit(X)
            for i in range(len(X)):
                if (s[i] not in union) and (s[i] in self.w2v.wv.vocab):
                    dct[kmedoids.labels_[i]].append(s[i])
            len_dct = 0
            seed = []
            # print(dct)
            for k, v in dct.items():
                if len(v) > 0:
                    len_dct += 1
                    seed.append(" ".join(v))
            seeds.extend(seed)
            dcts = [len_dct] * len(seed)
            len_dcts.extend(dcts)
            num_clusters.append(len_dct)
        return seeds, num_clusters, len_dcts

    def sent2idx(self, tokens: List[str], is_sent):
        if is_sent:
            tokens = [
                self.w2id[token.strip()]
                for token in tokens
                if token in self.w2id.keys()
            ]
            tokens += [self.pad_idx] * (self.maxlen - len(tokens))
            tokens = tokens[: self.maxlen]
        else:
            tokens = [
                self.w2id[token.strip()]
                for token in tokens
                if token in self.w2id.keys()
            ]
            ratio = int(self.maxlen_seed / len(tokens)) + 1
            tokens = tokens * ratio
            tokens = tokens[: self.maxlen_seed]
        return tokens

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, id):
        sent = self.sent2idx(self.test_data[id][0].split(), 1)
        self.mask = [i != self.pad_idx for i in sent]
        if np.array(self.mask).max() == 0:
            self.mask = np.concatenate(([1], np.zeros(len(self.mask) - 1)))
        seeds = []
        for item in self.all_seed:
            seeds.append(self.sent2idx(item.split(), 0))
            idx_sd = self.sent2idx(item.split(), 0)
        label = self.test_data[id][1]
        return (
            torch.LongTensor(sent),
            torch.LongTensor(seeds),
            torch.as_tensor(label),
            torch.as_tensor(self.num_clusters),
            torch.as_tensor(self.num_arr),
            torch.as_tensor(np.array(self.mask).astype(int)),
        )
