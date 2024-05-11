import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from modules import UNet


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def plot_images(images: torch.Tensor):
    """
    Create a small image grid from a tensor and plots them.

    Args:
        images (torch.Tensor): the tensor containing the images
    """
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images: torch.Tensor, path: str, **kwargs):
    """
    Create a small image grid from a tensor and saves it to the disk.

    Args:
        images (torch.Tensor): the tensor containing the images
        path (str): path to save the images to
    """
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args: dict) -> DataLoader:
    """
    Creates a `DataLoader` from a directory containing any images.

    Args:
        args (dict): the run arguments

    Returns:
        DataLoader: a DataLoader containing the training images ready to be fed to the model
    """
    transforms = torchvision.transforms.Compose([
        # args.image_size + 1/4 *args.image_size
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(
            args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(
        args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def save_image_grid(dir_path: str, img_size: int = 64, grid_name: str = "samples.jpg") -> None:
    """
    Create a collage of images located in a directory and saves it in the same directory.

    Args:
        dir_path (str): path to the directory containing the images
        img_size (int, optional): size (in pixels) of the images. Defaults to 64.
        grid_name (str, optional): file name under which the final grid will be saved. Defaults to "samples.jpg".
    """
    image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
    num_images = len(image_files)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    grid_width = num_cols * img_size
    grid_height = num_rows * img_size
    grid = Image.new('RGB', (grid_width, grid_height))
    x, y = 0, 0
    for image_file in image_files:
        img = Image.open(os.path.join(dir_path, image_file))
        grid.paste(img, (x, y))
        x += img_size
        if x >= grid_width:
            x = 0
            y += img_size
    grid.save(os.path.join(dir_path, grid_name))


def save_checkpoint(model: UNet, optimizer: torch.optim.Optimizer, save_path: str, epoch: int) -> None:
    """
    Saves the state of a model and the optimizer to make it possible to resume training from the last saved checkpoint.

    Args:
        model: model to save
        optimizer: optimizer used to train the model
        save_path: where to store the checkpoint
        epoch: epoch at which the checkpoint is saved

    Source: https://stackoverflow.com/a/63655261/22117082
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, os.path.join(save_path, f"{epoch+1}.pt"))


def load_checkpoint(model: UNet, optimizer: torch.optim.Optimizer, model_path: str):
    """
    Loads a saved checkpoint to resume training of a model.

    Args:
        model (UNet): model in which the weights have to be loaded
        optimizer (torch.optim.Optimizer): optimizer used to train the model
        model_path (str): where the checkpoint is saved

    Returns:
        saved model, optimizer and epoch

    Source: https://stackoverflow.com/a/63655261/22117082
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def load_last_checkpoint(model: UNet, optimizer: torch.optim.Optimizer, dir_path: str):
    """
    Loads the last checkpoint from a training process.

    Args:
        model (UNet): model in which the weights have to be loaded
        optimizer (torch.optim.Optimizer): optimizer used to train the model
        dir_path (str): path to the directory where the models where saved during the training process

    Returns:
        saved model, optimizer and epoch
    """
    if not os.path.exists(dir_path):
        return model, optimizer, 0
    available_pt_files = os.listdir(dir_path)
    if not available_pt_files:
        return model, optimizer, 0
    last_epoch = int(
        max(available_pt_files, key=lambda x: int(x.split('.')[0])).split('.')[0])
    print(f"Loading {last_epoch}.pt...")
    try:
        return load_checkpoint(model, optimizer, os.path.join(dir_path, f"{last_epoch}.pt"))
    except RuntimeError:
        if 1 < last_epoch:
            print(
                f"Could not load {last_epoch}.pt, most likely due to a corrupted checkpoint file, trying to load {last_epoch-1}.pt instead.")
            return load_checkpoint(model, optimizer, os.path.join(dir_path, f"{last_epoch-1}.pt"))
        else:
            print(
                f"Could not load {last_epoch}.pt, most likely due to a corrupted checkpoint file, starting training process from scratch.")
            return model, optimizer, 0


def setup_logging(run_name: str):
    """
    Initializes the necessary directories before a run.

    Args:
        run_name (str): name of the run
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name,
                "training_progress"), exist_ok=True)
