from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from torch.utils.data import random_split
from model.net import Model
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import torch
from torch import nn
from dataset import TrainDataset, TestDataset
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR


class Trainer(object):
    def __init__(self, args, device):
        super(Trainer, self).__init__()
        self.device = device
        self.w2v = Word2Vec.load(args.w2v)
        if "<pad>" not in self.w2v.wv.vocab:
            self.w2v.wv.add("<pad>", np.zeros(200, dtype=np.float))
        if args.pretrain_model != None:
            print("Found pretrained. Loading model...", end=" ")
            self.model = self.load_model(args.pretrain_model).to(self.device)
            print("Done")
        else:
            self.model = Model(args, self.w2v).to(self.device)
        self.optm = self.configure_optimizers(args)
        self.scheduler = StepLR(self.optm, step_size=1, gamma=1)

    def configure_optimizers(self, args):
        return torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)

    def prepare_data(self, args):
        data = TrainDataset(args, self.w2v)
        a = int(0.9 * len(data))
        b = len(data) - a
        self.train_ds, self.val_ds = random_split(
            data, [a, b], generator=torch.Generator().manual_seed(42)
        )

    def prepare_test_data(self, args):
        self.test_ds = TestDataset(args, self.w2v)

    def train_dataloader(self, args):
        return data.DataLoader(
            self.train_ds, batch_size=args.batchsize, num_workers=3, shuffle=True
        )

    def val_dataloader(self, args):
        return data.DataLoader(
            self.val_ds, batch_size=args.batchsize, num_workers=3, shuffle=True
        )

    def test_dataloader(self, args):
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=args.batchsize, num_workers=1, shuffle=False
        )

    def training_step(self, batch):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        sent, seeds, num_clusters, num_arr, mask = batch
        sent = sent.to(self.device)
        seeds = seeds.to(self.device)
        num_clusters = num_clusters.to(self.device)
        num_arr = num_arr.to(self.device)
        mask = mask.to(self.device)
        loss = self.model(sent, seeds, num_clusters, num_arr, mask, 1)
        loss.backward()
        self.optm.step()
        self.optm.zero_grad()

        return {"loss": loss.detach()}

    def training_epoch_end(self, outputs):
        """for each epoch end

        Arguments:
            outputs: list of training_step output
        """
        loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        return loss_mean

    def val_step(self, batch):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        sent, seeds, num_clusters, num_arr, mask = batch
        sent = sent.to(self.device)
        seeds = seeds.to(self.device)
        num_clusters = num_clusters.to(self.device)
        num_arr = num_arr.to(self.device)
        mask = mask.to(self.device)
        with torch.no_grad():
            loss = self.model(sent, seeds, num_clusters, num_arr, mask, 1)
        return {"loss": loss.detach()}

    def test_step(self, batch):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        sent, seeds, label, num_clusters, num_arr, mask = batch
        sent = sent.to(self.device)
        seeds = seeds.to(self.device)
        label = label.to(self.device)
        num_arr = num_arr.to(self.device)
        num_clusters = num_clusters.to(self.device)
        mask = mask.to(self.device)
        prd = []
        lb = []

        with torch.no_grad():
            preds = self.model(
                sent, seeds, num_clusters, num_arr, mask, 0
            )  # [batchsize]
        for pred, lbl in zip(preds, label):
            prd.append(pred.tolist())
            lb.append(lbl.tolist())
        return {"prd": prd, "lb": lb}

    def test_epoch_end(self, outputs):
        pred = []
        label = []
        for x in outputs:
            pred.extend(x["prd"])
        for x in outputs:
            label.extend(x["lb"])

        fscore = f1_score(label, pred, average="micro")
        precision = precision_score(label, pred, average="micro")
        recall = recall_score(label, pred, average="micro")
        acc = accuracy_score(label, pred)
        print(classification_report(label, pred, digits=4))
        results = {
            "fscore": fscore,
            "precision": precision,
            "recall": recall,
            "acc": acc,
        }
        return results

    def fit(self, args):
        self.prepare_data(args)
        train_dl = self.train_dataloader(args)
        val_dl = self.val_dataloader(args)
        loss_plt = []
        val_loss = []
        f1 = []
        acc = []
        for epoch in range(args.n_epoch):
            n_epc = args.n_epoch
            print(f"Epoch {epoch}/{n_epc-1}:", end=" ")
            outputs = []
            self.model.train()
            for batch in train_dl:
                # print(f'Training on batch {batch_id+1}/{len(train_dl)}')
                output = self.training_step(batch)
                outputs.append(output)
            loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
            print("Loss: {:0.4f}\n".format(loss_mean.item(), end="\n"))
            loss_plt.append(loss_mean)
            self.model.eval()
            loss_valid = []
            for batch in val_dl:
                loss_val = self.val_step(batch)
                loss_valid.append(loss_val)
            loss_val_mean = torch.stack([x["loss"] for x in loss_valid]).mean()
            print("Loss val: {:0.4f}\n".format(loss_val_mean.item(), end="\n"))
            val_loss.append(loss_val_mean)
            fscore = self.test(args)
            f1.append(fscore)
            if fscore > args.threshold:
                print(fscore)
                self.save_model(fscore, epoch + 1, args)
            self.scheduler.step()
        print("f1-micro: ", f1)
        plt.plot(loss_plt, label="train loss")
        plt.plot(val_loss, label="val loss")
        plt.plot(f1, label="f1score test")
        plt.legend(loc="best")
        plt.show()

    def test(self, args):
        print("Testing on", self.device)
        self.prepare_test_data(args)
        test_dl = self.test_dataloader(args)
        print("Testing...")
        t1 = time.time()
        outputs = []
        self.model.eval()
        for batch in test_dl:
            output = self.test_step(batch)
            outputs.append(output)
        results = self.test_epoch_end(outputs)
        t2 = time.time()
        print("Test time: {:0.2f}s".format(t2 - t1))
        if args.save_initmodel:
            torch.save(self.model, "/content/pretrain")
        return results["fscore"]

    def save_model(self, fscore, epoch, args):
        torch.save(
            self.model,
            f"{args.path_save_model}{args.des}fscore={fscore}epoch={epoch}use_att_sent={args.use_att_sent}use_att_seed_teacher={args.use_att_seed_teacher}use_att_seed_student={args.use_att_seed_student}",
        )

    def load_model(self, pretrain_model):
        return torch.load(pretrain_model)
