import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from model.layers import Embeddings, SentencesEmbeddings, Normalize
import numpy as np

np.random.seed(0)


class Model(nn.Module):
    def __init__(
        self,
        args,
        w2v,
        # word_emb_dropout, word_emb_froze,use_att
    ):
        super(Model, self).__init__()
        self.w2v = w2v
        self.word_emb_dropout = args.word_emb_dropout
        self.word_emb_froze_teacher = args.word_emb_froze_teacher
        self.word_emb_froze_student = args.word_emb_froze_student
        self.w2v_dim = self.w2v.wv.vector_size
        self.use_att_sent = args.use_att_sent
        self.use_att_seed_teacher = args.use_att_seed_teacher
        self.use_att_seed_student = args.use_att_seed_student
        self.maxlen = args.maxlen
        self.maxlen_seed = args.maxlen_seed
        self.weight_teacher = args.weight_teacher
        self.weight_student = args.weight_student
        self.word_emb_teacher = Embeddings(
            self.w2v, args, self.word_emb_froze_teacher, self.weight_teacher
        )
        self.word_emb_student = Embeddings(
            self.w2v, args, self.word_emb_froze_student, self.weight_student
        )
        self.sent_emb = SentencesEmbeddings(
            args, self.use_att_sent, self.w2v.wv.vector_size
        )
        self.sd_eb_teacher = SentencesEmbeddings(
            args, self.use_att_seed_teacher, self.w2v.wv.vector_size
        )
        self.sd_eb_student = SentencesEmbeddings(
            args, self.use_att_seed_student, self.w2v.wv.vector_size
        )
        self.number_loss = args.number_loss
        self.normal = Normalize()
        self.anpha = args.anpha
        self.lamda = args.lamda
        self.des = args.des
        self.beta = args.beta
        self.is_train = args.is_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, sents, seeds, num_clusters, num_arr, mask, flag):

        # ground-truth
        snt_emb_teacher = self.sent_emb(self.word_emb_teacher(sents), mask)
        sd_emb_teacher = self.sd_eb_teacher(self.word_emb_teacher(seeds))

        snt_emb_teacher_transpose = torch.transpose(snt_emb_teacher.unsqueeze(1), 1, 2)
        pro_pred_teacher = torch.matmul(
            sd_emb_teacher, snt_emb_teacher_transpose
        ).squeeze(-1)

        # uce
        snt_emb_student = self.sent_emb(self.word_emb_student(sents), mask)
        sd_emb_student = self.sd_eb_student(self.word_emb_student(seeds))
        snt_emb_student_transpose = torch.transpose(snt_emb_student.unsqueeze(1), 1, 2)
        pro_pred_student = torch.matmul(
            sd_emb_student, snt_emb_student_transpose
        ).squeeze(-1)

        # ------cluster---------------
        tmp = num_clusters[0]
        pro_tmp = torch.max(pro_pred_student[:, : tmp[0]], -1).values.unsqueeze(-1)
        index_max = torch.argmax(pro_pred_student[:, : tmp[0]], -1).unsqueeze(-1)
        for i in range(1, tmp.shape[0]):
            tmp_ = torch.max(
                pro_pred_student[:, torch.sum(tmp[:i]) : torch.sum(tmp[: i + 1])], -1
            ).values.unsqueeze(-1)
            index = torch.argmax(
                pro_pred_student[:, torch.sum(tmp[:i]) : torch.sum(tmp[: i + 1])], -1
            ).unsqueeze(-1)
            pro_tmp = torch.cat((pro_tmp, tmp_), 1)
            index_max = torch.cat((index_max, index), 1)
        pro_pred_student = pro_tmp
        seed_emb = torch.take(sd_emb_student, index_max)

        pro_tmp = torch.max(pro_pred_teacher[:, : tmp[0]], -1).values.unsqueeze(-1)
        for i in range(1, tmp.shape[0]):
            tmp_ = torch.max(
                pro_pred_teacher[:, torch.sum(tmp[:i]) : torch.sum(tmp[: i + 1])], -1
            ).values.unsqueeze(-1)
            pro_tmp = torch.cat((pro_tmp, tmp_), 1)
        pro_pred_teacher = pro_tmp
        # -------END--------

        # predict
        predict_teacher, reli_teacher = self.normal(pro_pred_teacher)
        predict_student, reli_student = self.normal(pro_pred_student)

        if flag:
            loss = self.my_loss(predict_teacher, pro_pred_student, reli_teacher)
            return loss
        else:
            if self.des == "Citysearch":
                pro_pred_student = pro_pred_student[:, :3]  # for CitySearch Corpus
            return torch.argmax(pro_pred_student, -1)
            # return torch.argmax(pro_pred_teacher,-1)

    def my_loss(self, predict_teacher, pro_pred_student, reli_teacher):
        tmp_denta = predict_teacher - pro_pred_student
        if self.number_loss == 0:
            loss = torch.mul(
                (1 + self.anpha * torch.abs(reli_teacher)),
                torch.mul(tmp_denta, tmp_denta),
            ).sum()
        loss = loss / predict_teacher.shape[0]
        return loss
