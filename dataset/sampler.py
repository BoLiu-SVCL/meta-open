import torch
import numpy as np


class MetaSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, opts_runtime, train=True):
        super(MetaSampler, self).__init__(dataset)

        self.iterations = opts_runtime.iterations
        self.n_way = opts_runtime.n_way
        self.k_shot = opts_runtime.k_shot
        self.m_query = opts_runtime.m_query
        self.p_base = opts_runtime.p_base
        self.open_cls = opts_runtime.open_cls
        self.open_sample = opts_runtime.open_sample
        self.fold = opts_runtime.fold
        self.train = train

        self.n_sample_list = dataset.n_sample_list
        if train:
            self.n_cls = dataset.cls_num
            self.base_cls = 0
            self.idx_list = []
            for i in range(self.n_cls):
                self.idx_list.append(np.arange(self.n_sample_list[:i].sum(), self.n_sample_list[:i+1].sum()))
        else:
            self.n_cls = dataset.open_cls_num
            self.base_cls = dataset.cls_num
            self.idx_list = []
            for i in range(dataset.cls_num):
                self.idx_list.append(np.arange(self.n_sample_list[:i].sum(), self.n_sample_list[:i+1].sum()))

    def __iter__(self):
        for it in range(self.iterations):
            batch_s = torch.zeros(self.n_way * self.k_shot)
            batch_q = torch.zeros(self.n_way * self.m_query)
            batch_open = torch.zeros(self.open_cls * self.open_sample)

            cls_all = torch.from_numpy(np.random.permutation(self.n_cls))
            cls_fsl = cls_all[:self.n_way]
            cls_open = cls_all[self.n_way: self.n_way + self.open_cls]
            for c in range(self.n_way):
                n_sample = int(self.n_sample_list[cls_fsl[c]+self.base_cls].item())
                samples = np.random.permutation(n_sample)[:self.k_shot + self.m_query]
                supports = samples[:self.k_shot]
                querys = samples[self.k_shot:]
                batch_s[self.k_shot*c:self.k_shot*(c+1)] = torch.from_numpy(supports) + self.n_sample_list[:cls_fsl[c]+self.base_cls].sum()
                batch_q[self.m_query*c:self.m_query*(c+1)] = torch.from_numpy(querys) + self.n_sample_list[:cls_fsl[c]+self.base_cls].sum()
            for c in range(self.open_cls):
                n_sample = int(self.n_sample_list[cls_open[c]+self.base_cls].item())
                samples = np.random.permutation(n_sample)[:self.open_sample]
                batch_open[self.open_sample*c:self.open_sample*(c+1)] = torch.from_numpy(samples) + self.n_sample_list[:cls_open[c]+self.base_cls].sum()
            idx_all = np.concatenate(self.idx_list)
            np.random.shuffle(idx_all)
            if self.train:
                batch_e = torch.from_numpy(idx_all[:self.p_base]).float()
                batch = torch.cat((batch_s, batch_q, batch_open, batch_e), dim=0).long().view(-1)
                yield batch
            else:
                if self.fold > 1:
                    fold_q = int(self.n_way * self.m_query / self.fold)
                    fold_open = int(self.open_cls * self.open_sample / self.fold)
                    for i in range(self.fold):
                        batch_q_fold = batch_q[fold_q*i:fold_q*(i+1)]
                        batch_open_fold = batch_open[fold_open*i:fold_open*(i+1)]
                        batch = torch.cat((batch_s, batch_q_fold, batch_open_fold), dim=0).long().view(-1)
                        yield batch
                else:
                    batch = torch.cat((batch_s, batch_q, batch_open), dim=0).long().view(-1)
                    yield batch

    def __len__(self):
        return self.iterations * self.fold
