import os
import torch
from torch.optim import Adam
from torch import nn
import argparse
from argparse import ArgumentParser
from trainer import Trainer
from dataset import TrainDataset, TestDataset
from model.net import Model
from model.layers import Embeddings, SentencesEmbeddings


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    parser = ArgumentParser("train args")
    parser.add_argument("--des", default=None, type=str, help="description")
    parser.add_argument("--n_epoch", default=30, type=int, help="number of epochs")
    parser.add_argument("--train_data", default=None, type=str, help="train data")
    parser.add_argument("--test_data", default=None, type=str, help="test data")
    parser.add_argument("--val_data", default=None, type=str, help="validation data")
    parser.add_argument("--seed_data", default=None, type=str, help="seed data")
    parser.add_argument("--w2v", default=None, type=str, help="path of word2v model")
    parser.add_argument(
        "--maxlen", default=25, type=int, help="max length of sentences"
    )
    parser.add_argument(
        "--maxlen_seed", default=30, type=int, help="max length of seed words"
    )
    parser.add_argument("--batchsize", default=50, type=int, help="batch_size")
    parser.add_argument("--start_epoch", default=0, type=int, help="epoch count from")
    parser.add_argument(
        "--word_emb_dropout", default=0, type=float, help="word embedding dropout"
    )
    parser.add_argument(
        "--word_emb_froze_teacher",
        default=True,
        type=bool,
        help="word embedding froze for teacher",
    )
    parser.add_argument(
        "--word_emb_froze_student",
        default=False,
        type=bool,
        help="word embedding froze for student",
    )
    parser.add_argument(
        "--use_att_seed_student",
        default=0,
        type=int,
        help="Use attention for seeds or not for teacher",
    )
    parser.add_argument(
        "--use_att_seed_teacher",
        default=0,
        type=int,
        help="Use attention for seeds or not for student",
    )
    parser.add_argument(
        "--use_att_sent", default=0, type=int, help="Use attention for sentences or not"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--anpha",
        default=0,
        type=float,
        help="anpha hyparameter for loss function - student",
    )
    parser.add_argument(
        "--beta",
        default=0,
        type=float,
        help="anpha hyparameter for loss function - teacher",
    )
    parser.add_argument(
        "--weight_teacher",
        default=None,
        type=str,
        help="pretrain weight for tearcher embedding",
    )
    parser.add_argument(
        "--weight_student",
        default=None,
        type=str,
        help="pretrain weight for student embedding",
    )
    parser.add_argument(
        "--lamda", default=0, type=float, help="lamda hyparameter for loss function"
    )
    parser.add_argument(
        "--path_save_model", default="/content/", type=str, help="path to save model"
    )
    parser.add_argument(
        "--pretrain_model", default=None, type=str, help="path to pretrain model"
    )
    parser.add_argument("--is_train", default=0, type=int, help="Train or Test")
    parser.add_argument(
        "--threshold", default=0.83, type=float, help="threshold to save model"
    )
    parser.add_argument(
        "--save_initmodel", default=0, type=int, help="to save init model"
    )
    parser.add_argument(
        "--number_loss", default=0, type=int, help="To choose the loss function"
    )
    parser.add_argument(
        "--phi", default=0, type=float, help="hyparameter for loss function"
    )
    parser.add_argument(
        "--low_prior", default=0, type=float, help="hyparameter for loss function"
    )
    parser.add_argument(
        "--high_prior", default=1, type=float, help="hyparameter for loss function"
    )
    parser.add_argument(
        "--num_cluster_general",
        default=1,
        type=int,
        help="number cluster for general seed words",
    )
    parser.add_argument(
        "--num_cluster_aspects",
        default=3,
        type=int,
        help="number cluster for seed words",
    )
    parser.add_argument(
        "--general_idx", default=-1, type=int, help="index of general labels"
    )
    parser.add_argument(
        "--maxlen_seed_aspects",
        default=3,
        type=int,
        help="number seedwords for aspects != general",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    trainer = Trainer(args, device)
    if args.is_train:
        trainer.fit(args)
    else:
        trainer.test(args)
