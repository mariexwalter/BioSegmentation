"""
https://github.com/akshaykulkarni07/pl-sem-seg/blob/master/pl_training.ipynb

https://pytorch-lightning.readthedocs.io/en/1.6.1/common/loggers.html

# TODO: test data?

"""
from argparse import ArgumentParser
from os.path import join

import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim
import torchmetrics
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

import augmentation
import datasets


class BinarySegmentation(pl.LightningModule):
    def __init__(self, conf):
        super(BinarySegmentation, self).__init__()
        self.setup_neural_net(conf)
        self.setup_datasets(conf)
        self.setup_training(conf)
        self.save_hyperparameters()

    def setup_training(self, conf):
        self.batch_size = conf.training.batch_size
        self.learning_rate = conf.training.learning_rate
        self.loss = nn.CrossEntropyLoss(weight=conf.training.class_weights)
        self.opt = conf.optimizer.name
        self.opt_kwargs = (
            conf.optimizer.kwargs if conf.optimizer.kwargs is not None else {}
        )
        self.example_input_array = torch.zeros(*conf.dataset.input_shape)

        self.metrics = {
            'acc': torchmetrics.Accuracy('binary', num_classes=2),
            'ji': torchmetrics.JaccardIndex('binary', zero_division=.1),
        }

        self.add_im_every_n_steps = conf.log.add_image_every_n_steps

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label = batch
        img = img.float()
        label = label.long()[:, 0, ...]
        pred = self.forward(img)
        loss_val = self.loss(pred, label)

        if batch_idx % self.add_im_every_n_steps == 0:
            self.logger.experiment.add_image(
                "input", img[0, ...], self.global_step
            )
            self.logger.experiment.add_image(
                "label", label[0, ...].unsqueeze(0), self.global_step
            )
            self.logger.experiment.add_image(
                "pred", pred[0, 1, ...].unsqueeze(0), self.global_step
            )

        out = {'loss': loss_val}

        for key, fcn in self.metrics.items():
            out[key] = fcn.to(pred.device)(pred[:, 1, ...], label)

        self.log_dict(
            out, on_step=False, on_epoch=True, prog_bar=True
        )

        return out

    def validation_step(self, batch, batch_idx):
        img, label = batch
        img = img.float()
        label = label.long()[:, 0, ...]
        pred = self.forward(img)
        loss_val = self.loss(pred, label)

        out = {'val_loss': loss_val}

        for key, fcn in self.metrics.items():
            out['val_'+key] = fcn.to(pred.device)(pred[:, 1, ...], label)

        self.log_dict(
            out, on_step=False, on_epoch=True, prog_bar=True
        )

        return out

    def test_step(self, batch, batch_idx):
        img, label = batch
        img = img.float()
        label = label.long()[:, 0, ...]
        pred = self.forward(img)
        loss_val = self.loss(pred, label)

        out = {'test_loss': loss_val}

        for key, fcn in self.metrics.items():
            out['test_'+key] = fcn.to(pred.device)(pred[:, 1, ...], label)

        self.log_dict(
            out, on_step=False, on_epoch=True, prog_bar=True
        )

        return out

    def configure_optimizers(self):
        opt_cls = torch.optim.__dict__[self.opt]
        opt = opt_cls(
            self.net.parameters(),
            lr=self.learning_rate,
            **self.opt_kwargs
        )
        # so far no scheduling
        sch = None
        return [opt]

    def train_dataloader(self):
        return DataLoader(
            self.data['train'], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data['test'], batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data['test'], batch_size=self.batch_size, shuffle=False
        )

    def setup_datasets(self, conf):
        ds_class = datasets.__dict__[conf.dataset.name]
        augment_fcn = augmentation.__dict__[conf.dataset.augmentation.name]
        self.data = {}
        for train_or_test in ['train', 'test']:
            self.data[train_or_test] = ds_class(
                augment_fcn('train', **conf.dataset.augmentation),
                conf.dataset[train_or_test].images.path,
                conf.dataset[train_or_test].images.ext,
                conf.dataset[train_or_test].labels.path,
                conf.dataset[train_or_test].labels.ext,
            )

    def setup_neural_net(self, conf):
        # see https://github.com/qubvel/segmentation_models.pytorch#architectures

        model_cls = smp.__dict__[conf.neuralnet.name]
        self.net = model_cls(
            # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name=conf.neuralnet.encoder_name,
            # use `imagenet` pre-trained weights for encoder initialization
            encoder_weights=conf.neuralnet.encoder_weights,
            # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            in_channels=conf.neuralnet.in_channels,
            # model output channels (number of classes in your dataset)
            classes=conf.neuralnet.classes,
        )


def main():
    # get sys args
    options = get_sys_args()
    # load config
    conf = OmegaConf.load(join(options.exp_folder, "config.yaml"))
    seed_everything(conf.training.seed, workers=True)
    model = BinarySegmentation(conf)

    tb_logger = TensorBoardLogger(
        options.exp_folder, name=conf.log.name,
        log_graph=True,
    )
    csv_logger = CSVLogger(join(options.exp_folder))

    trainer = Trainer(
        accelerator=options.accelerator,
        devices=[options.devices],
        max_epochs=conf.training.epochs,
        log_every_n_steps=conf.log.log_every_n_steps,
        logger=[tb_logger, csv_logger],
        # profiler="simple"
    )
    trainer.fit(
        model,
        model.train_dataloader(),
        model.val_dataloader(),
    )
    trainer.test(dataloaders=model.test_dataloader())


def get_sys_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_folder", type=str)
    parser.add_argument("--accelerator", default="cuda")
    parser.add_argument("--devices", default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
