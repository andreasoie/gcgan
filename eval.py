import os
import subprocess
import warnings

import numpy as np
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions
from util.visualizer import save_image

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.no_flip = True
    opt.display_id = -1 

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataloader = dataset.dataloader
    opt.results_dir += f"/{opt.model}"
    os.makedirs(opt.results_dir, exist_ok=True)
    resume_epochs = np.linspace(25, 400, 16, dtype=int).tolist()
    
    for resume_epoch in tqdm(resume_epochs, desc="Epoch", total=len(resume_epochs), leave=False):
        opt.epoch = resume_epoch
        model = create_model(opt)      # create a model given opt.model and other options
        model.set_eval()
        
        OUTDIR = os.path.join(opt.results_dir, f"step_{resume_epoch}")
        os.makedirs(OUTDIR, exist_ok=True)
        
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating epoch {resume_epoch}", leave=False):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()   # get image results
            img_paths = model.get_image_paths()     # get image paths
            save_image(OUTDIR, visuals, img_paths)
        del model
        
        # Eval FID
        REAL_IMG_PATH = "/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry/testB"
        BATCH_SIZE_FID = 9
        
        command = ['python', 'fid.py', '--paths', REAL_IMG_PATH, OUTDIR, '--img_size', str(256), '--batch_size', str(BATCH_SIZE_FID)]
        result = subprocess.run(command, stdout=subprocess.PIPE)
        score = float(result.stdout.decode('utf-8').split(" ")[-1])
        
        println = f"FID [{str(resume_epoch).rjust(3)}]: {score}"
        print(println)
        
        with open(f"fid_{opt.model}.txt", "a") as f:
            f.write(f"{println} \n")
        
        # Remove OUTDIR
        os.system(f"rm -rf {OUTDIR}")
