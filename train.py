from collections import OrderedDict
import time

import numpy as np
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt

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


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
opt.max_iter_num = (opt.niter + opt.niter_decay)*3000


def save_snapshot_image(idx: int, visuals: OrderedDict) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=len(visuals), squeeze=False, figsize=(40, 10))
    for i, (label, image) in enumerate(visuals.items()):
        image = tensor2im(image)
        axs[0, i].imshow(image)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_title(label, fontsize=25)
    plt.savefig(f"tmp/step_{idx}.png")
    plt.tight_layout()
    plt.close()
    
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch}"):
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
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if total_steps % opt.save_latest_freq == 0:
            # print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')
            
        if total_steps % 2500 == 0:
            model.set_eval()
            with torch.no_grad():
                model.test()
                save_snapshot_image(total_steps, model.get_current_visuals())
            model.set_train()

    if epoch % opt.save_epoch_freq == 0:
        # print('Saving the model at epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
