"""
https://github.com/akshaykulkarni07/pl-sem-seg/blob/master/pl_training.ipynb

"""
from argparse import ArgumentParser
from os.path import join

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
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

    def setup_training(self, conf):
        self.batch_size = conf.training.batch_size
        self.learning_rate = conf.training.learning_rate
        self.loss = nn.CrossEntropyLoss(weight=conf.training.class_weights)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        pred = self.forward(img)
        loss_val = self.loss(pred, mask[:, 0, ...])
        return {'loss': loss_val}

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate
        )
        # so far no scheduling
        sch = None
        return [opt]

    def train_dataloader(self):
        return DataLoader(
            self.data['train'], batch_size=self.batch_size, shuffle=True
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

    trainer = Trainer(
        accelerator=options.accelerator,
        devices=[options.devices],
        max_epochs=conf.training.epochs,
        log_every_n_steps=conf.log.log_every_n_steps
    )
    trainer.fit(model)


def get_sys_args():
    parser = ArgumentParser()
    parser.add_argument("--exp_folder", type=str)
    parser.add_argument("--accelerator", default="cuda")
    parser.add_argument("--devices", default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
