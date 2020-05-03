
1. Unpack dataset into "data" folder.
2. Run `sh bin/prepare_data.sh` 
3. Run `catalyst-dl run -C configs/_common.yml configs/main.yml --logdir=baseline`


#### Data structure
Make sure, that final folder with data has the required structure:
```bash
/path/to/your_dataset/
        class_name_1/
            images
        class_name_2/
            images
        ...
        class_name_100500/
            ...
```
#### Data location

* The easiest way is to move your data:
    ```bash
    mv /path/to/your_dataset/* /catalyst.classification/data/origin
    ```
    In that way you can run pipeline with default settings.

* If you prefer leave data in `/path/to/your_dataset/`
    * In local environment:
        * Link directory
            ```bash
            ln -s /path/to/your_dataset $(pwd)/data/origin
            ```
         * Or just set path to your dataset `DATADIR=/path/to/your_dataset` when you start the pipeline.

    * Using docker

        You need to set:
        ```bash
           -v /path/to/your_dataset:/data \ #instead default  $(pwd)/data/origin:/data
         ```
        in the script below to start the pipeline.


## 3. Classification pipeline
### Fast&Furious: raw data → production-ready model

The pipeline will automatically guide you from raw data to the production-ready model.

We will initialize ResNet-18 model with a pre-trained network. During current pipeline model will be trained sequentially in two stages, also in the first stage we will train several heads simultaneously.

#### Run in local environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDNN_BENCHMARK="True" \
CUDNN_DETERMINISTIC="True" \
bash ./bin/catalyst-classification-pipeline.sh \
  --workdir ./logs \
  --datadir ./data/origin \
  --max-image-size 224 \  # 224 or 448 works good
  --balance-strategy 256 \  # images in epoch per class, 1024 works good
  --config-template ./configs/templates/main.yml \
  --num-workers 4 \
  --batch-size 256 \
  --criterion CrossEntropyLoss \  # one of CrossEntropyLoss, BCEWithLogits, FocalLossMultiClass
```

#### Run in docker:

```bash
export LOGDIR=$(pwd)/logs
docker run -it --rm --shm-size 8G --runtime=nvidia \
   -v $(pwd):/workspace/ \
   -v $LOGDIR:/logdir/ \
   -v $(pwd)/data/origin:/data \
   -e "CUDA_VISIBLE_DEVICES=0" \
   -e "USE_WANDB=1" \
   -e "LOGDIR=/logdir" \
   -e "CUDNN_BENCHMARK='True'" \
   -e "CUDNN_DETERMINISTIC='True'" \
   -e "WORKDIR=/logdir" \
   -e "DATADIR=/data" \
   -e "MAX_IMAGE_SIZE=224" \
   -e "BALANCE_STRATEGY=256" \
   -e "CONFIG_TEMPLATE=./configs/templates/main.yml" \
   -e "NUM_WORKERS=4" \
   -e "BATCH_SIZE=256" \
   -e "CRITERION=CrossEntropyLoss" \
   catalyst-classification ./bin/catalyst-classification-pipeline.sh
```
The pipeline is running and you don’t have to do anything else, it remains to wait for the best model!

#### Visualizations

You can use [W&B](https://www.wandb.com/) account for visualisation right after `pip install wandb`:

```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
```
<img src="/pics/wandb_metrics.png" title="w&b classification metrics"  align="left">

Tensorboard also can be used for visualisation:

```bash
tensorboard --logdir=/catalyst.classification/logs
```
<img src="/pics/tf_metrics.png" title="tf classification metrics"  align="left">

<details>
<summary>Confusion matrix</summary>
<p>
<img src="/pics/cm.png" title="tf classification metrics" width="700">
</p>
</details>

## 4. Results
All results of all experiments can be found locally in `WORKDIR`, by default `catalyst.classification/logs`. Results of experiment, for instance `catalyst.classification/logs/logdir-191010-141450-c30c8b84`, contain:

#### checkpoints
*  The directory contains all checkpoints: best, last, also of all stages.
* `best.pth` and `last.pht` can be also found in the corresponding experiment in your W&B account.

#### configs
*  The directory contains experiment\`s configs for reproducibility.

#### logs
* The directory contains all logs of experiment.
* Metrics also logs can be displayed in the corresponding experiment in your W&B account.

#### code
*  The directory contains code on which calculations were performed. This is necessary for complete reproducibility.

## 5. Customize own pipeline

For your future experiments framework provides powerful configs allow to optimize configuration of the whole pipeline of classification in a controlled and reproducible way.

<details>
<summary>Configure your experiments</summary>
<p>

* Common settings of stages of training and model parameters can be found in `catalyst.classification/configs/_common.yml`.
    * `model_params`: detailed configuration of models, including:
        * model, for instance `MultiHeadNet`
        * detailed architecture description
        * using pretrained model
    * `stages`: you can configure training or inference in several stages with different hyperparameters. In our example:
        * optimizer params
        * first learn the head(s), then train the whole network

* The `CONFIG_TEMPLATE` with other experiment\`s hyperparameters, such as data_params and is here: `catalyst.classification/configs/templates/main.yml`.  The config allows you to define:
    * `data_params`: path, batch size, num of workers and so on
    * `callbacks_params`: Callbacks are used to execute code during training, for example, to get metrics or save checkpoints. Catalyst provide wide variety of helpful callbacks also you can use custom.


You can find much more options for configuring experiments in [catalyst documentation.](https://catalyst-team.github.io/catalyst/)

</p>
</details>
