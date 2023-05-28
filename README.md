## Links

Cell dataset:

https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation?resource=download

U-net models:

https://github.com/qubvel/segmentation_models.pytorch

Augmentation:

https://pytorch.org/vision/0.15/transforms.html

## important commands

### running the training
python main.py --exp_folder experiments/test_01

### to visualize the training
tensorboard --logdir experiments/

### change port
--port 6009

### to predict images without labels
python main.py --predict --checkpoint_path path/to/exp_folder
