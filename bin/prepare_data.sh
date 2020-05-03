DATASET_DIR=data/train


catalyst-data tag2label \
    --in-dir ${DATASET_DIR} \
    --out-dataset ${DATASET_DIR}/dataset_raw.csv \
    --out-labeling ${DATASET_DIR}/tag2class.json

catalyst-data split-dataframe \
    --in-csv ${DATASET_DIR}/dataset_raw.csv \
    --tag2class ${DATASET_DIR}/tag2class.json \
    --tag-column=tag --class-column=class \
    --n-folds=5 --train-folds=0,1,2,3 \
    --out-csv=${DATASET_DIR}/dataset.csv


DATASET_DIR=data/test


catalyst-data tag2label \
    --in-dir ${DATASET_DIR} \
    --out-dataset ${DATASET_DIR}/dataset_raw.csv \
    --out-labeling ${DATASET_DIR}/tag2class.json

