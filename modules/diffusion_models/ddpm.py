import logging
import math
import os

import fire
import torch
import torch.nn as nn
from pytorch_fid import fid_score
from pytorch_gan_metrics import get_inception_score_from_directory
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import *

from modules import UNet

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, show_progress: bool = True):
        model.eval()
        with torch.no_grad():
            x = torch.randn(
                (n, 3, self.img_size, self.img_size)).to(self.device)
            iterator = tqdm(reversed(range(1, self.noise_steps)),
                            desc=f"Sampling {n} images", unit="step") if show_progress else reversed(range(1, self.noise_steps))
            for i in iterator:
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat)))
                                             * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args: dotdict):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(img_size=args.image_size, device=args.device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    last_epoch = 0
    if args.resume:
        model, optimizer, last_epoch = load_last_checkpoint(
            model, optimizer, f"./models/{args.run_name}/")
        logging.info(f"Resuming training at epoch {last_epoch+1}.")
        last_epoch += 1
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    for epoch in range(last_epoch, args.epochs):
        pbar = tqdm(
            dataloader, desc=f"epoch {epoch+1}/{args.epochs}", unit="batch")
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        save_checkpoint(model, optimizer, f"./models/{args.run_name}", epoch)


def launch_training(
    run_name: str = "untitled",
    dataset: str = "./datasets/MNIST/train",
    epochs: int = 500,
    batch_size: int = 1,
    img_size: int = 64,
    device: str = "cuda",
    lr: float = 3e-4,
    resume: bool = True
):
    """
    Launches a training session.
    Train a new model. In the future, it will allows to continue training an existing one.

    Args:
        dataset (str): path to the directory containing the dataset.
        run_name (str, optional): name to give to the run. Defaults to "untitled".
        epochs (int, optional): epoch count. Defaults to 500.
        batch_size (int, optional): size of the batches. Defaults to 1.
        img_size (int, optional): size of the images (in pixels), must be a MULTIPLE OF 8. Defaults to 64.
        device (str, optional): device on which pytorch will be running. Defaults to "cuda".
        lr (float, optional): learning rate of the model. Defaults to 3e-4.
        resume (bool, optional): if True and a checkpoint is saved, will resume training from last checkpoint. Defaults to True.
    """
    if img_size % 8 != 0:
        prev_img_size = img_size
        img_size = math.ceil((img_size+7)//8) * 8
        logging.warning(
            f"The img_size you provided ({prev_img_size}) is not a multiple of 8. It has been rounded to the closest greater multiple of 8 ({img_size}).")
    args = dotdict({
        "dataset_path": dataset,
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": img_size,
        "device": device,
        "lr": lr,
        "resume": resume
    })
    train(args)


def launch_sampling(
    n_img: int = 1,
    run_name: str = "untitled",
    model_path: str = None,
    name: str = "samples",
    device: str = "cuda",
    img_size: int = 64,
    save_grid: bool = False
):
    """
    Samples images using a model trained previously.

    Args:
        n_img (int, optional): how many images should be sampled. Defaults to 1.
        run_name (str, optional): name of the run, shoud be the one used when training the model. Defaults to "untitled".
        model_path (str, optional): Path of the checkpoint from which images should be sampled. If left to none, will use the model at ./models/run_name/checkpoint.pt. Defaults to None.
        name (str, optional): name of the folder in which the sampled images will be saved. Defaults to "samples".
        device (str, optional): device on which pytorch will be running. Defaults to "cuda".
        img_size (int, optional): size of the images to sample, should be size used when training the model, must be a MULTIPLE OF 8. Defaults to 64.
        save_grid (bool, optional): whether to save a collage of the samples at the end or not. Defaults to False.
    """
    if img_size % 8 != 0:
        prev_img_size = img_size
        img_size = math.ceil((img_size+7)//8) * 8
        logging.warning(
            f"The img_size you provided ({prev_img_size}) is not a multiple of 8. It has been rounded to the closest greater multiple of 8 ({img_size}).")
    model = UNet(img_size=img_size).to(device)
    model_path = model_path or os.path.join(
        "./models/", run_name, "checkpoint.pt")
    model, _, _ = load_checkpoint(
        model, optim.AdamW(model.parameters()), model_path)
    diffusion = Diffusion(img_size=img_size, device=device)
    dir_path = f"./results/{run_name}/{name}/"
    os.makedirs(dir_path, exist_ok=True)
    for i in tqdm(range(n_img), desc=f"Sampling {n_img} images", unit="image", position=0):
        sampled_images = diffusion.sample(model, 1, show_progress=False)
        save_images(sampled_images, dir_path+f"sample_{i+1}.jpg")
    if save_grid:
        save_image_grid(dir_path, img_size)
    logging.info(f"Sampled {n_img} images, saved at {dir_path}")


def launch_evalutation(
    models_path: str = "./models/untitled/",
    dataset: str = "./datasets/untitled/test",
    epoch_frequency: int = 10,
    max_epoch: int = 500,
    img_count: int = 1000,
    device: str = "cuda",
    img_size: int = 64,
    output_samples_path: str = "./samples/evaluation/untitled/",
    output_csv_path: str = "./evaluations/untitled.csv"
):
    """
    Start the evaluation of trained models, computing the Inception Score (IS) and Fréchet Inception Distance (FID) of each model on the test set.
    The report is then saved as a csv file storing: epoch, IS, FID

    Args:
        models_path (str, optional): path to the directory where the trained models are located. Models should be named as they were during training, i.e. "xxx.pt" where xxx is the epoch. Defaults to "./models/untitled/".
        dataset (str, optional): test dataset on which the scores should be computed. Defaults to "./datasets/untitled/test".
        epoch_frequency (int, optional): frequency at which the models should be evaluated, e.g. if 10, the models 10.pt, 20.pt, 30.pt... will be evaluated. Defaults to 10.
        max_epoch (int, optional): total number of epochs on which the model was trained. Defaults to 500.
        img_count (int, optional): number of samples to generate for evaluation. Defaults to 1000.
        device (str, optional): device on which pytorch will be running. Defaults to "cuda".
        img_size (int, optional): size of the images to sample, should be size used when training the model, must be a MULTIPLE OF 8. Defaults to 64.
        output_samples_path (str, optional): path to the directory where samples generated for evaluation will be saved. Defaults to "./samples/evaluation/untitled/".
        output_csv_path: (str, optional): file path where to save the resulting CSV. If the file already exists it will be overwritten. Defaults to "./evaluation/untitled.csv"
    """
    with open(output_csv_path, "w+") as result_file:
        result_file.write("epoch, IS, FID\n")
    model = UNet(img_size=img_size).to(device)
    for epoch in range(1, max_epoch+1):
        if epoch % epoch_frequency == 0 or epoch in (1, max_epoch):
            # Load the model checkpoint of this epoch and use it to sample img_count images
            checkpoint_path = os.path.join(models_path, f"{epoch}.pt")
            model, _, _ = load_checkpoint(model, optim.AdamW(
                model.parameters()), checkpoint_path)
            diffusion = Diffusion(img_size=img_size, device=device)
            out_dir = os.path.join(output_samples_path, str(epoch))
            os.makedirs(out_dir, exist_ok=True)
            for i in tqdm(range(img_count), desc=f"Sampling for epoch {epoch}", unit=" image"):
                sample = diffusion.sample(model, 1, show_progress=False)
                save_images(sample, os.path.join(out_dir, f"sample_{i+1}.jpg"))
            # Compute IS
            inception_score = get_inception_score_from_directory(
                out_dir, use_torch=True)[0]
            # Compute FID
            fid = fid_score.calculate_fid_given_paths(
                [dataset, out_dir], dims=2048, batch_size=256, device=device)
            # Append resulting line to CSV file
            with open(output_csv_path, "a") as result_file:
                result_file.write(f"{epoch}, {inception_score}, {fid}\n")
    logging.info("Done evaluating the models.")


if __name__ == '__main__':
    fire.Fire({
        "train": launch_training,
        "sample": launch_sampling,
        "evaluate": launch_evalutation
    })
