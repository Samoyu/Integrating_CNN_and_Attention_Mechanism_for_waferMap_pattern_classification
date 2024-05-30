import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils import convert_gray_to_rgb
from torchvision import transforms
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.features = dataframe['waferMap'].values
        self.labels = dataframe['failureType'].values
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        sample = Image.fromarray(sample.astype(np.uint8))

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.features)
    

def get_dataset(size, bs, seed, augment=False):
    print('========================== Start Loading Dataset ==========================')
    if augment:
        dataset = pd.read_pickle("dataset_aug.pkl")
    else:
        dataset = pd.read_pickle("dataset.pkl")

    # Preprocess the dataset
    dataset['waferMap'] = dataset['waferMap'].apply(convert_gray_to_rgb)
    print(f"Shape of wafermap is : {dataset.loc[0]['waferMap'].shape}")
    mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,
                  'Random':5,'Scratch':6,'Near-full':7,'none':8}
    dataset = dataset.replace({'failureType':mapping_type})
    dataset = dataset[(dataset['failureType']>=0) & (dataset['failureType']<=7)].reset_index()
    print(f'type of dataset is : {type(dataset)}')
    dataset = dataset.loc[0:500]
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=seed)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(dataset_train, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataset = CustomDataset(dataset_test, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    print('========================== Finish Loading Dataset ==========================')
    return train_dataloader, test_dataloader