import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import json

from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss

device = torch.device("cuda")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Function to calculate accuracy
def binary_accuracy(preds, y):
    # preds are the raw scores output by the model
    # Convert these scores to predicted class indices
    preds_class = preds.argmax(dim=1)
    # Compare the predicted classes with the true classes
    correct = (preds_class == y).float()
    # Calculate the accuracy
    acc = correct.sum() / len(correct)
    return acc

def best_cv_val(nested_list):
    array = np.array(nested_list)
    means = np.mean(array, axis=0)
    best_epoch = np.argmax(means)
    return best_epoch + 1, means[best_epoch]

class RandomGaussianBlur:
    def __init__(self, kernel_size=3, probability=0.5):
        self.kernel_size = kernel_size
        self.probability = probability
        self.gaussian_blur = transforms.GaussianBlur(self.kernel_size)

    def __call__(self, img):
        if random.random() < self.probability:
            return self.gaussian_blur(img)
        return img

def split_dataset(holdout_clusters, full_dataset):
    # Determine the validation indices by checking if the file matches any holdout cluster
    val_indices = [i for i, (path, _) in enumerate(full_dataset.imgs)
                   if any(f"_{cluster}.png" in path for cluster in holdout_clusters)]

    # The training set includes all indices that are not in the validation set
    train_indices = [i for i in range(len(full_dataset)) if i not in val_indices]

    return train_indices, val_indices

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

os.chdir('/home/kdoherty/spurge/data_release')

train_dir = './data/crop_39/train'

df = pd.read_csv('./results/best_lr.csv')
best_row = df.loc[df['accuracy'].idxmax()]
learning_rate = best_row['learning_rate']

with open('./results/best_augs.json', 'r') as file:
    augs = json.load(file)

gaussian_blur = augs['gaussian_blur']
flip_horizontal = augs['flip_horizontal']
flip_vertical = augs['flip_vertical']
brightness = augs['brightness']
contrast = augs['contrast']
saturation = augs['saturation']
hue = augs['hue']
rotation = augs['rotation']

with open('./results/best_cutmix.json', 'r') as file:
    cutmix_results = json.load(file)

cutmix_prob = cutmix_results["cutmix_prob"]
cutmix_beta = cutmix_results["cutmix_beta"]
cutmix_num_mix = cutmix_results["cutmix_num_mix"]

stats = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
transform_list = [transforms.ToTensor(), stats]

if gaussian_blur:
    transform_list.insert(0, RandomGaussianBlur())

if flip_horizontal:
    transform_list.insert(0, transforms.RandomHorizontalFlip())

if flip_vertical:
    transform_list.insert(0, transforms.RandomVerticalFlip())

transform_list.insert(0, transforms.ColorJitter(hue=hue, contrast=contrast, brightness=brightness, saturation=saturation))

transform_list.insert(0, transforms.RandomRotation(rotation))

train_transforms = transforms.Compose(transform_list)
val_transforms = transforms.Compose([transforms.ToTensor(), stats])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(train_dir, transform=val_transforms)

seed = 0

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

g = torch.Generator()
g.manual_seed(seed)

batch_size = 32
n_epochs = 500
holdout_sets = [[0], [1], [2], [4], [5], [6,7], [8]]

array_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
holdout_set = holdout_sets[array_idx]

print(f'Validating clusters {holdout_set}')

# Now split the dataset
train_indices, val_indices = split_dataset(holdout_set, train_dataset)

# Create subsets
train_subset = Subset(train_dataset, train_indices)
train_subset = CutMix(train_subset, num_class=2, beta=cutmix_beta, prob=cutmix_prob, num_mix=cutmix_num_mix)

val_subset = Subset(val_dataset, val_indices)

# Create the data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Load pre-trained resnet50 model + higher level layers
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)

criterion = CutMixCrossEntropyLoss(True).to(device)

model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

epochs = range(n_epochs)
epoch_accs = []
epoch_losses = []

with tqdm(total=n_epochs*len(train_loader), unit="batch", desc="Training Progress") as pbar:
    for epoch in epochs:
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            output = model(images)
            output = output.unsqueeze(0) if len(output.shape) == 1 else output
            loss = criterion(output, labels.unsqueeze(0) if labels.dim() == 0 else labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.update(1)
        
        train_loss = running_loss / len(train_loader)
        
        # Validate the model
        model.eval()
        running_loss = 0
        running_acc = 0
        
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            output = model(images)
            output = output.unsqueeze(0) if len(output.shape) == 1 else output
            loss = criterion(output, labels.unsqueeze(0) if labels.dim() == 0 else labels)
            acc = binary_accuracy(output, labels.float())
            running_loss += loss.item()
            running_acc += acc.item()

        val_loss = running_loss/len(val_loader)
        val_acc = running_acc/len(val_loader)
        epoch_accs.append(val_acc)
        epoch_losses.append(val_loss)
        
        pbar.set_postfix({'Epoch': epoch, 
                          'Validation Loss': f'{val_loss:.3f}', 
                          'Validation Accuracy': f'{val_acc:.3f}'})

os.makedirs('./results/epoch_tune_cutmix', exist_ok=True)

# Convert the lists to a DataFrame
results_df = pd.DataFrame({
    'array_idx': [holdout_set] * n_epochs,
    'epoch': list(epochs),
    'epoch_loss': epoch_losses,
    'epoch_acc': epoch_accs
})

# Save to CSV
csv_path = f'./results/epoch_tune_cutmix/{holdout_set}.csv'
results_df.to_csv(csv_path, index=False)