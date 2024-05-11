import os
import shutil

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


def save_dataset(dataset: Dataset, dest_path: str, split_classes: bool = True) -> None:
    """
    Converts image tensors to png files and saves them to the corresponding label folder.

    Args:
        dataset (Dataset): torch dataset to save to the disk.
        dest_path (str): path to the destination folder.
        split_classes (bool, optional): weither to split the images between folders named according to their classes or not. Defaults to True.
    """
    # convert images tensors to png and save them to the corresponding label folder.
    for i, (image, label) in tqdm(enumerate(dataset), desc="Converting dataset", unit=" images"):
        if split_classes:
            img_dir = os.path.join(dest_path, str(
                label).strip().replace("\n", ""))
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
        else:
            img_dir = dest_path
        image = transforms.ToPILImage()(image)
        image = image.convert()
        image_filename = os.path.join(img_dir, f"{i}.png")
        image.save(image_filename)


def safe_rm(path: str) -> None:
    """
    Safely removes a file or a directory, handling the case where is does not exists.

    Args:
        path (str): path of the file or directory to remove.
    """
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


# instancing of the pytorch datasets to be saved
transform = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(
    "./datasets/MNIST/train/",
    train=True,
    download=True,
    transform=transform
)
mnist_test = datasets.MNIST(
    "./datasets/MNIST/test/",
    train=False,
    download=True,
    transform=transform
)

cifar10_train = datasets.CIFAR10(
    "./datasets/CIFAR10/train/",
    train=True,
    download=True,
    transform=transform
)
cifar10_test = datasets.CIFAR10(
    "./datasets/CIFAR10/test/",
    train=False,
    download=True,
    transform=transform
)

print("\nProcessing MNIST...")
save_dataset(mnist_train, "./datasets/MNIST/train/")
save_dataset(mnist_test, "./datasets/MNIST/test/", split_classes=False)
print("\nProcessing CIFAR-10...")
save_dataset(cifar10_train, "./datasets/CIFAR10/train/")
save_dataset(cifar10_test, "./datasets/CIFAR10/test/", split_classes=False)

print("\nRemoving raw files...")
safe_rm("./datasets/MNIST/train/MNIST")
safe_rm("./datasets/MNIST/test/MNIST")
safe_rm("./datasets/CIFAR10/train/cifar-10-batches-py")
safe_rm("./datasets/CIFAR10/test/cifar-10-batches-py")
safe_rm("./datasets/CIFAR10/train/cifar-10-python.tar.gz")
safe_rm("./datasets/CIFAR10/test/cifar-10-python.tar.gz")

print("\nDataset conversion done, images saved at ./datasets/.")
