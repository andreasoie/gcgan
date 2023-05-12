MODE=test
VBS=1
BS=1

CHECKPOINT_DIR=/home/andy/Dropbox/largefiles1/logs/gcgan_bs8/checkpoints
VALDIR="/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry"
EVAL_DIR="inferences"

python eval.py \
--dataroot $VALDIR \
--name rgb2ir \
--model gc_gan_cross \
--checkpoints_dir $CHECKPOINT_DIR \
--results_dir $EVAL_DIR \
--no_dropout \
--loadSize 256 \
--which_model_netG resnet_6blocks \
--which_direction AtoB \
--dataset_mode unaligned \
--resize_or_crop "resize" \
--batchSize 8 \
--nThreads 4 \
--input_nc 3 \
--output_nc 3 \
--gpu_ids 0 \
--geometry rot