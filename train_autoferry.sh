DATAROOT="/home/andy/data/complete_dataset_processed/autoferry"

python train.py --dataroot $DATAROOT \
--name rgb2ir \
--model gc_gan_cross \
--pool_size 50 \
--no_dropout \
--loadSize 256 \
--which_model_netG resnet_6blocks \
--batchSize 1 \
--input_nc 3 \
--output_nc 3 \
--gpu_ids 0 \
--identity 0.3 \
--which_direction AtoB \
--geometry rot \
--resize_or_crop "resize" \
--dataset_mode unaligned \
--no_html