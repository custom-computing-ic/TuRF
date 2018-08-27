#!/bin/sh

START_INDEX=$1
NUM_EPOCHS=150

python src/train_fine_tune_models.py \
  --trainable_scopes "vgg_16/conv1/conv1_1,vgg_16/conv1/conv1_2,
                      vgg_16/conv2/conv2_1,vgg_16/conv2/conv2_2,
                      vgg_16/conv3/conv3_1,vgg_16/conv3/conv3_2,vgg_16/conv3/conv3_3,
                      vgg_16/conv4/conv4_1,vgg_16/conv4/conv4_2,vgg_16/conv4/conv4_3,
                      vgg_16/conv5/conv5_1,vgg_16/conv5/conv5_2,vgg_16/conv5/conv5_3,
                      vgg_16/fc6,vgg_16/fc7,vgg_16/fc8" \
  --num_epochs $NUM_EPOCHS \
  --fine_tuning_mode iterative \
  --start_index $START_INDEX
