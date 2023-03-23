import os
import pdb
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import wandb
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_snapshot_image(fname: str, visuals: OrderedDict) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=len(visuals), squeeze=False, figsize=(40, 10))
    for i, (label, image) in enumerate(visuals.items()):
        image = tensor2im(image)
        axs[0, i].imshow(image)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_title(label, fontsize=25)
    plt.savefig(fname)
    plt.tight_layout()
    plt.close()
    
def seconds_to_time(seconds: float) -> str:
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    seconds = seconds - hours * 3600 - minutes * 60
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
# pin memory to speed up data loading
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

print('N Training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
opt.max_iter_num = (opt.niter + opt.niter_decay)*3000

NAME = "gcgan"

os.makedirs(f"snapshots/{NAME}", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

wandb.init(project="gcgan", entity="andreasoie", resume="allow")
wandb.config.update(opt, allow_val_change=True)

round_trip_times = []
n_epochs_left = opt.niter + opt.niter_decay - opt.epoch_count
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch}", leave=False):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        opt.iter_num = total_steps
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            meta = {"epoch": epoch, "epoch_iter": epoch_iter, **errors}
            wandb.log(meta)
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if total_steps % opt.save_latest_freq == 0:
            for save_path in model.save('latest'):
                wandb.save(save_path)
                
        if total_steps % 2500 == 0:
            model.set_eval()
            with torch.no_grad():
                model.test()
                filename = f"snapshots/step_{total_steps}.png"
                save_snapshot_image(filename, model.get_current_visuals())
                wandb.log({"snapshot": wandb.Image(filename)})
            model.set_train()

    if epoch % opt.save_epoch_freq == 0:
        for save_path in model.save('latest'):
            wandb.save(save_path)
        for save_path in model.save(epoch):
            wandb.save(save_path)

    lr = model.update_learning_rate()
    n_epochs_left -= 1
    round_trip_time = time.time() - epoch_start_time
    round_trip_times.append(round_trip_time)
    avg_time_left = seconds_to_time(n_epochs_left * (np.mean(round_trip_times)))
    print(f"End of epoch {str(epoch).ljust(3)}/{str(opt.niter + opt.niter_decay).ljust(3)}, Time Taken: {str(round(round_trip_time, 1)).ljust(5)} sec, lr = {str(lr).ljust(10)}, AVG ETR: {avg_time_left.ljust(13)}")
