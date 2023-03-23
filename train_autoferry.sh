DATAROOT="/home/andy/data/complete_dataset_processed/autoferry"

python train.py --dataroot $DATAROOT \
--name rgb2ir \
--model gc_gan_cross \
--pool_size 50 \
--no_dropout \
--loadSize 256 \
--which_model_netG resnet_6blocks \
--which_direction AtoB \
--dataset_mode unaligned \
--resize_or_crop "resize" \
--batchSize 1 \
--nThreads 4 \
--input_nc 3 \
--output_nc 3 \
--gpu_ids 0 \
--identity 0.3 \
--geometry rot \
--no_html \
--niter 200 \
--niter_decay 200