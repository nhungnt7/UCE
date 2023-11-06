import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init


class Embeddings(nn.Module):
    def __init__(self, w2v, args, word_emb_froze, weight=None):
        super(Embeddings, self).__init__()
        # super().__init__()
        if "<pad>" in w2v.wv.vocab:
            self.pad_idx = w2v.wv.vocab.get("<pad>").index
        else:
            self.pad_idx = w2v.wv.vocab.get("etc").index

        if weight == None:
            self.weight = w2v.wv.vectors
        else:
            self.weight = torch.load(weight)
        self.word_emb_froze = word_emb_froze
        self.word_emb = nn.Embedding.from_pretrained(
            torch.as_tensor(self.weight),
            freeze=self.word_emb_froze,
            padding_idx=self.pad_idx,
        )
        self.word_emb_dropout = nn.Dropout(args.word_emb_dropout)
        self.output_dim = w2v.vector_size

    def forward(self, words):
        # word = [batchsize, n_sample, maxlen]
        embedding_out = self.word_emb_dropout(self.word_emb(words))
        return embedding_out


class SentencesEmbeddings(nn.Module):
    def __init__(self, args, use_att, w2v_dim: int):
        super(SentencesEmbeddings, self).__init__()
        self.use_att = use_att
        # print(args.use_att)
        self.w2v_dim = w2v_dim
        self.maxlen = args.maxlen

    def forward(self, sent, mask=None):

        if self.use_att == 0:
            flag = len(sent.shape) - 2
            # if mask != None:
            #     return torch.sum(sent, flag)/torch.sum(mask, -1).unsqueeze(-1)
            return sent.mean(axis=flag)

        else:
            if len(sent.shape) == 3:
                att_weights = self.att(sent)
                sent_emb = torch.transpose(sent, 1, 2)
                sent_eb = torch.matmul(sent_emb, att_weights.unsqueeze(2))
                sent_eb = sent_eb.squeeze(2)
                return sent_eb
            else:
                # sent [batchsize,n_labels, maxlen, w2v_dim]
                sents = sent.reshape(-1, sent.shape[2], sent.shape[3])
                att_weights = self.att(sents)
                att_weights = att_weights.reshape(
                    sent.shape[0], sent.shape[1], sent.shape[2]
                )
                sent_eb = torch.matmul(att_weights.unsqueeze(2), sent)
                sent_eb = sent_eb.squeeze(2)
                return sent_eb


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, pro_pred):
        # predict
        tmp_pro = torch.argmax(pro_pred, dim=-1)
        predict = torch.nn.functional.one_hot(
            tmp_pro, pro_pred.shape[-1]
        )  # [bz, n_labels]
        # reli
        tmp_reli = pro_pred - torch.min(pro_pred, dim=-1).values.unsqueeze(1)
        tmp_reli = tmp_reli / (
            (torch.max(tmp_reli, dim=-1).values.unsqueeze(1)) + 0.001
        )
        reli = torch.abs(1 - predict - tmp_reli)  # [bz, n_labels]
        return predict, reli
