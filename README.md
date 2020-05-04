## Catalyst classification example with training and infer stage

### Installation:
To install all requirements run `pip install -r requirements/requirements.txt`

### Training
1. Unpack dataset into "data" folder. In my example I use 2 classes. You should have next structure.
```
data
├── test
    ├── 0001.jpg
    ├── 0002.jpg 
    └── ... 
└── train
    ├── 0_class
        ├── 0001.jpg
        ├── 0002.jpg 
        └── ... 

    ├── 1_class
        ├── 0001.jpg
        ├── 0002.jpg 
        └── ... 
    ...

    ├── n_class
        ├── 0001.jpg
        ├── 0002.jpg 
        └── ... 
```

2. Rename folders in train to 0,1,... and so on. 
```
data
├── test
    ├── 0001.jpg
    ├── 0002.jpg 
    └── ... 
└── train
    ├── 0
        ├── 0001.jpg
        ├── 0002.jpg 
        └── ... 

    ├── 1
        ├── 0001.jpg
        ├── 0002.jpg 
        └── ... 
    ...

    ├── n_class
        ├── 0001.jpg
        ├── 0002.jpg 
        └── ... 
```
3. Run `sh bin/prepare_data.sh` to prepare data. It splits you train folder into train and val and preprocess test for infer stage.
4. Run `catalyst-dl run -C configs/_common.yml configs/main.yml --logdir=baseline` If you want use more than 2 classes change `&num_classes 2` to custom number in infer.yml and main.yml files. (Important! Not set to 1 class, this feature isn't supported)
5. Run `catalyst-dl run -C configs/_common.yml configs/infer.yml --logdir=baseline --logdir=baseline --autoresume=best` to make predictions. It shows prediction for each file and dump it into 'infer_pred.txt' file in 'baseline' folder
