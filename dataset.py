import glob
import os 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None, domain_label=1):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform
        self.domain_label = int(domain_label)
        if not self.files:
            raise RuntimeError(f"No DeepGlobe images found in:\n{root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.domain_label


class DomainLabelDataset(Dataset):
    def __init__(self, dataset, domain_label):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.domain_label

def build_dataloaders(data_root_e, data_root_d, batch_size, num_workers, device, fixed_count=16):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    base_e = datasets.ImageFolder(root=data_root_e, transform=transform)
    base_e = Subset(base_e, range(min(50000, len(base_e))))
    dataset_e = DomainLabelDataset(base_e, domain_label=0)
    dataset_d = SatelliteDataset(data_root_d, transform=transform, domain_label=1)
    dataset_d = Subset(dataset_d, range(min(50000, len(dataset_d))))
    options = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
                   pin_memory=(device == "cuda"), persistent_workers=(num_workers > 0),
                   drop_last=True)
    loader_e = DataLoader(dataset_e, **options)
    loader_d = DataLoader(dataset_d, **options)
    if len(loader_e) == 0 or len(loader_d) == 0:
        raise ValueError("Each dataset must contain at least batch_size images (drop_last=True).")
    count = min(fixed_count, len(dataset_e), len(dataset_d))
    fixed_e = torch.stack([dataset_e[i][0] for i in range(count)]).to(device)
    fixed_d = torch.stack([dataset_d[i][0] for i in range(count)]).to(device)
    return dataset_e, dataset_d, loader_e, loader_d, fixed_e, fixed_d