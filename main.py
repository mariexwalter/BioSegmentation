"""
# TODO: Resume training
# TODO: Scheduling
# TODO: predict image -> Tiling might not work properly
"""
import math
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from os.path import basename, join, splitext

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim
import torchmetrics
import torchvision.transforms.functional as TF
from blended_tiling import TilingModule
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

import augmentation
import datasets


class BinarySegmentation(pl.LightningModule):
    def __init__(self, conf):
        super(BinarySegmentation, self).__init__()
        self.setup_neural_net(conf)
        self.setup_datasets(conf)
        self.setup_training(conf)
        self.setup_prediction(conf)

        self.save_hyperparameters()

    def setup_training(self, conf):
        # For some reason, there are problems with seeding and using xent-loss, because a subfunction is not deterministic
        # This is a work-around proposed in:
        # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/8
        xent = nn.CrossEntropyLoss(weight=conf.training.class_weights, reduction="none")
        def loss(x,y):
            loss = xent(x,y)
            return loss.mean()
        
        self.loss = loss
        self.batch_size = conf.training.batch_size
        self.learning_rate = conf.training.learning_rate
        self.opt = conf.optimizer.name
        self.opt_kwargs = (
            conf.optimizer.kwargs if conf.optimizer.kwargs is not None else {}
        )
        if "scheduler" in conf.optimizer.keys():
            self.sched = conf.optimizer.scheduler    
            self.sched_kwargs = (
                conf.optimizer.sched_kwargs if conf.optimizer.sched_kwargs is not None else {}
            )
        else:
            self.sched = None    

        self.example_input_array = torch.zeros(*conf.dataset.input_shape)

        self.metrics = {
            'acc': torchmetrics.Accuracy('binary', num_classes=2),
            'ji': torchmetrics.JaccardIndex('binary', zero_division=.1),
        }

        self.add_im_every_n_steps = conf.log.add_image_every_n_steps

    def setup_prediction(self, conf):
        self.tile_size = conf.predict.tile_size
        self.tile_overlap = conf.predict.tile_overlap

    def forward(self, x):
        return self.net(x)

    def predict_with_tiling(self, image):
        softmax = nn.Softmax(dim=1)
        image = TF.to_tensor(image).float()

        tiling_module = TilingModule(
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            base_size=image.shape[-2:],
        )

        tiles = tiling_module.split_into_tiles(image.unsqueeze(0))

        num_images = int(math.ceil(tiles.shape[0] / self.batch_size))

        out = []
        for i in range(num_images):
            batch = tiles[i*self.batch_size:(i+1)*self.batch_size, ...]
            if batch.shape[0] == 0:
                break

            with torch.no_grad():
                out.append(
                    softmax(self.forward(batch))[:, 1, ...].unsqueeze(1)
                )

        out = torch.cat(out, 0)
        full_tensor = tiling_module.rebuild_with_masks(out)
        # return as binary image
        return full_tensor[0, 0, ...].cpu().numpy()

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

        if self.sched is not None:
            # could be, e.g., MultiStepLR
            scheduler_cls = torch.optim.lr_scheduler.__dict__[self.sched]
            scheduler = scheduler_cls(opt, **self.sched_kwargs)
            return [opt], [scheduler]

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


def load_model(path_to_checkpoint):
    model = BinarySegmentation.load_from_checkpoint(path_to_checkpoint)
    return model.eval()


def process_images(model, conf, exp_folder):
    load_fcn = datasets.__dict__[conf.dataset.predict.load_fcn]
    to_search = glob(
        join(conf.dataset.predict.path, '*'+conf.dataset.predict.ext)
    )
    assert to_search

    pred_folder = join(exp_folder, 'predictions')
    makedirs(pred_folder, exist_ok=True)

    for image_path in tqdm(to_search):
        name = splitext(basename(image_path))[0]
        image = load_fcn(image_path)
        segmentation = model.predict_with_tiling(image)
        segmentation = Image.fromarray(
            (250*segmentation).astype('uint8'), 'L'
        )
        segmentation.save(join(pred_folder, name + '.png'))


def main():
    # get sys args
    options = get_sys_args()
    # load config
    print(f"Trying to load { join(options.exp_folder, 'config.yaml') }")

    conf = OmegaConf.load(join(options.exp_folder, "config.yaml"))

    if options.checkpoint_path is not None:
        model = load_model(options.checkpoint_path)
    else:
        model = BinarySegmentation(conf)

    if options.predict:
        process_images(model, conf, options.exp_folder)
        return None

    seed_everything(conf.training.seed, workers=True)
    tb_logger = TensorBoardLogger(
        join(options.exp_folder, 'logs'), name='tb_logs',
        log_graph=True,
    )
    csv_logger = CSVLogger(join(options.exp_folder, 'logs'))

    trainer = Trainer(
        accelerator=options.accelerator,
        devices=[options.devices],
        max_epochs=conf.training.epochs,
        log_every_n_steps=conf.log.log_every_n_steps,
        logger=[tb_logger, csv_logger],
        deterministic=True,
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
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--accelerator", default="cuda")
    parser.add_argument("--devices", default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
