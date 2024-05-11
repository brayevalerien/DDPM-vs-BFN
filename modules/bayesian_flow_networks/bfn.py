import logging

import fire
import torch.nn as nn
import torch_ema
from model import BayesianFlow, UNet
from pytorch_fid import fid_score
from pytorch_gan_metrics import get_inception_score_from_directory
from torch.optim import AdamW
from tqdm.auto import tqdm
from utils import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


def train(args: dotdict):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(dropout_prob=.3, input_dim=3, output_dim=3).to(device)
    bfn = BayesianFlow(sigma=args.sigma)
    ema = torch_ema.ExponentialMovingAverage(model.parameters(), decay=0.9999)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      betas=(0.9, 0.98), weight_decay=0.01)
    last_epoch = 0
    if args.resume:
        model, optimizer, last_epoch = load_last_checkpoint(
            model, optimizer, f"./models/{args.run_name}/")
        logging.info(f"Resuming training at epoch {last_epoch+1}.")
    model.train()
    for epoch in range(last_epoch, args.epochs):
        pbar = tqdm(
            dataloader, desc=f"epoch {epoch+1}/{args.epochs}", unit=" images")
        for image, _ in pbar:
            optimizer.zero_grad()
            image = image.permute(0, 2, 3, 1).to(device)
            image = image*2 + 1
            loss = bfn.continuous_data_continuous_loss(model, image).loss
            pbar.set_postfix({'loss': loss.item()})
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
        save_checkpoint(model, optimizer, f"./models/{args.run_name}", epoch)


def launch_sampling(
    n_img: int = 1,
    steps: int = 100,
    sigma: float = .001,
    run_name: str = "untitled",
    model_path: str = None,
    name: str = "samples",
    device: str = "cuda",
    img_size: int = 64,
    save_grid: bool = False
) -> None:
    """
    Samples images using a previously trained BFN checkpoint.

    Args:
        n_img (int, optional): how many images should be sampled. Defaults to 1.
        steps (int, optional): number of sampling steps for the BFN. Defaults to 100.
        run_name (str, optional): name of the run, should be the one used when training the model. Defaults to "untitled".
        model_path (str, optional): path to the checkpoint from which images should be sampled. If left to None, will use the model at ./models/run_name/checkpoint. Defaults to None.
        name (str, optional): name of the folder in which the sampled images will be saved. Defaults to "samples".
        device (str, optional): device on which pytorch will be running. Defaults to "cuda".
        img_size (int, optional): size of the images to samples, should be the size used when training the models. Defaults to 64.
        save_grid (bool, optional): whether to save a collage of the samples at the end or not. Defaults to False.
    """
    model = UNet(dropout_prob=0, input_dim=3, output_dim=3).to(device)
    model_path = model_path or os.path.join(
        "./models/", run_name, "checkpoint.pt")
    model, _, _ = load_checkpoint(model, AdamW(model.parameters()), model_path)
    bfn = BayesianFlow(sigma=sigma)
    model.eval()
    dir_path = os.path.join("./results/", run_name, name)
    os.makedirs(dir_path, exist_ok=True)
    model.eval()
    for i in tqdm(range(n_img), desc=f"Sampling {n_img} images", unit="image"):
        sampled_images = bfn.continuous_data_sample(model, size=(
            1, img_size, img_size, 3), num_steps=steps, device=device)
        save_images(sampled_images[0], os.path.join(
            dir_path, f"sample_{i+1}.jpg"))
    if save_grid:
        save_image_grid(dir_path, img_size)
    logging.info(f"Sampled {n_img} images, saved at {dir_path}")


def launch_evalutation(
    models_path: str = "./models/untitled",
    dataset: str = "./datasets/untitled/test",
    epoch_frequency: int = 10,
    max_epoch: int = 500,
    sigma: float = .001,
    img_count: int = 1000,
    device: str = "cuda",
    img_size: int = 32,
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
    model = UNet(dropout_prob=0, input_dim=3, output_dim=3).to(device)
    for epoch in range(1, max_epoch+1):
        if epoch % epoch_frequency == 0 or epoch in (1, max_epoch):
            checkpoint_path = os.path.join(models_path, f"{epoch}.pt")
            model, _, _ = load_checkpoint(model, AdamW(
                model.parameters()), checkpoint_path)
            bfn = BayesianFlow(sigma=sigma)
            model.eval()
            out_dir = os.path.join(output_samples_path, str(epoch))
            os.makedirs(out_dir, exist_ok=True)
            for i in tqdm(range(img_count), desc=f"Sampling for epoch {epoch}", unit=" image"):
                sample = bfn.continuous_data_sample(model, size=(
                    1, img_size, img_size, 3), num_steps=100, device=device)
                save_images(sample, os.path.join(out_dir, f"sample_{i+1}.jpg"))
            inception_score = get_inception_score_from_directory(
                out_dir, use_torch=True)[0]
            # Compute FID
            fid = fid_score.calculate_fid_given_paths(
                [dataset, out_dir], dims=2048, batch_size=256, device=device)
            # Append resulting line to CSV file
            with open(output_csv_path, "a") as result_file:
                result_file.write(f"{epoch}, {inception_score}, {fid}\n")
    logging.info("Done evaluating the models.")


def launch_training(
    run_name: str = "untitled",
    dataset: str = "./datasets/MNIST/train",
    epochs: int = 500,
    sigma: float = .001,
    device: str = "cuda",
    lr: float = 3e-4,
    img_size: int = 32,
    resume: bool = True
) -> None:
    args = dotdict({
        "dataset_path": dataset,
        "run_name": run_name,
        "epochs": epochs,
        "device": device,
        "lr": lr,
        "image_size": img_size,
        "resume": resume,
        "batch_size": 1
    })
    train(args)


if __name__ == "__main__":
    fire.Fire({
        "train": launch_training,
        "sample": launch_sampling,
        "evaluate": launch_evalutation,
    })
