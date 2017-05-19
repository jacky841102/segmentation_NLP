# segmentation_NLP
This project is extended from [text_objseg](https://github.com/ronghanghu/text_objseg).

## Folder structure
```
/data                       #training data should be placed here
/log                        #checkpoints of tensorflow models and log files are saved here
/models                     #different model class
    /components             #convnet model (e.g. vgg, deeplab, resnet)
        /pretrained         #pretrained parms for convnet model
/util                       #util packages
```

### /data
training data should be placed here. e.g.
```
/data
    /referit
        /train_batch_seg
    /coco
        /train_batch_seg
    /coco+
        /train_batch_seg
    /cocoref
        /train_batch_seg
    ...
```

### /log
log dir will be created automatically once training a model

### /models
all models are placed here

All models inherit from `base` model, which implement common methods of models like `train()`, `initialize()`, `build_model()`, etc. Other models must inherit from `base`, and override `forward` method, which is the function defining the structure of the model.

### /models/components/pretrained
pretrained parms for convnet should be placed here. Dowload pretrained parms [here](https://drive.google.com/drive/folders/0B6CnOZnxTx5tLUhqNDJ5dUo0T1k?usp=sharing).

## Training
For example, to train the model based on FCN convnet, run the following code
```
from models.fcn import *
model = FCN()
model.train()
```

or, one can specify the parameters of the model while initializing
```
model = FCN(fix_convnet=False, vgg_dropout=True, max_iter=18000, start_lr=0.1)
```

The configurable parameters are not well-documented yet, and it can be found in `__init__` method of models now.

## Todo
- [] load model
- [] test model
- [x] train model
- [x] training summary
- [x] fcn-based model
- [x] deeplab-based model
- [] deeplab101-based model (use [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet))
- [] recurrent multimodal interaction model
- [x] integrate referit dataset
- [] integrate coco dataset
- [] integrate coco+ dataset
- [] integrate coco-ref dataset
- [] demo file
- [] document for model parameters
