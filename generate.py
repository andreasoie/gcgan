import json
import os

import numpy as np
from munch import Munch
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from fid import calculate_fid_given_paths
from models.gc_gan_cross_model import GcGANCrossModel
from models.models import create_model
from util.visualizer import save_images

gcgan_options = Munch({
    'dataroot': '/home/andreoi/data/study_cases',
    # "dataroot": "/home/andreoi/data/autoferry",
    "batchSize": 1,
    "batch_size_val": 9,
    "loadSize": 256,
    "fineSize": 256,
    "input_nc": 3,
    "output_nc": 3,
    "iter_num": 0,
    "ngf": 64,
    "ndf": 64,
    "loadSize": 256,
    "identity": 0.3,
    "no_lsgan": False,
    "which_model_netD": "basic",
    "which_model_netG": "resnet_6blocks",
    "n_layers_D": 3,
    "gpu_ids": [0],
    "lambda_A": 10,
    "lambda_B": 10,
    "pool_size": 50,
    "lambda_G": 1,
    "nThreads": 4,
    "name": "rgb2ir",
    "dataset_mode": "unaligned",
    "model": "gc_gan_cross",
    "which_direction": "AtoB",
    "nThreads": 1,
    "checkpoints_dir": "/home/andreoi/ckpts/autoferry_gcgan",
    "norm": "instance",
    "epoch_count": 0,
    "beta1": 0.5,
    "niter": 200,
    "niter_decay": 200,
    "serial_batches": False,
    "display_id": 0,
    "continue_train": False,
    "display_port": 8097,
    "no_dropout": True,
    "max_dataset_size": np.inf,
    "resize_or_crop": "resize",
    "no_flip": True,
    "init_type": "xavier",
    "ntest": np.inf,
    "results_dir": "generations",
    "aspect_ratio": 1.0,
    "phase": "test",
    "which_epoch": "latest",
    "how_many": 50,
    "geometry": "rot",
    "isTrain": False
})

        
if __name__ == "__main__":
    opt = gcgan_options
    
    dloader = CreateDataLoader(opt)
    valdataloader = dloader.dataloader
    
    assert len(dloader) == 6, "The number of dataloader should be 6, but it is {}".format(dloader)
    os.makedirs(opt.results_dir, exist_ok=True)
    
    resume_epochs = np.linspace(25, 400, 16).astype(int)

    for resume_epoch in tqdm(resume_epochs, desc="Epoch", total=len(resume_epochs), leave=False):
        opt.epoch = resume_epoch
        opt.which_epoch = resume_epoch
        model = create_model(opt)
        model.set_eval()
    
        tmpout = os.path.join(opt.results_dir, f"step_{resume_epoch}")
        os.makedirs(tmpout, exist_ok=True)
        
        for i, data in tqdm(enumerate(valdataloader), total=len(valdataloader), desc=f"Evaluating epoch {resume_epoch}", leave=False):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()   # get image results
            img_paths = model.get_image_paths()     # get image paths
            save_images(tmpout, visuals, img_paths)
        del model