import os
import torch
from pathlib import Path
from typing import List, Optional, Sequence, Union, Callable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule


# Stub for custom dataset if needed later
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


# Workaround to avoid CelebA integrity check errors
class MyCelebA(CelebA):
    def _check_integrity(self) -> bool:
        return True


# Optional OxfordPets dataset (not used here)
class OxfordPets(Dataset):
    def __init__(self, data_path: str, split: str, transform: Callable, **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, 0.0  # dummy label


# Main VAE DataModule
class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_name: str = "mnist",  # "celeba" or "mnist"
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_name = dataset_name.lower()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_name == "celeba":
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ])
            val_transforms = train_transforms

            self.train_dataset = MyCelebA(
                self.data_dir,
                split='train',
                transform=train_transforms,
                download=False,
            )
            self.val_dataset = MyCelebA(
                self.data_dir,
                split='test',
                transform=val_transforms,
                download=False,
            )

        elif self.dataset_name == "mnist":
            mnist_transforms = transforms.Compose([
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                # Uncomment below line if you want to convert grayscale to RGB:
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

            self.train_dataset = MNIST(
                root=self.data_dir,
                train=True,
                transform=mnist_transforms,
                download=True
            )
            self.val_dataset = MNIST(
                root=self.data_dir,
                train=False,
                transform=mnist_transforms,
                download=True
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
