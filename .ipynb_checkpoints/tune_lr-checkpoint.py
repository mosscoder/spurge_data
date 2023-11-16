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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def split_dataset(holdout_clusters, full_dataset):
    # Determine the validation indices by checking if the file matches any holdout cluster
    val_indices = [i for i, (path, _) in enumerate(full_dataset.imgs)
                   if any(f"_{cluster}.png" in path for cluster in holdout_clusters)]

    # The training set includes all indices that are not in the validation set
    train_indices = [i for i in range(len(full_dataset)) if i not in val_indices]

    return train_indices, val_indices

os.chdir('/home/kdoherty/spurge/data_release')
train_dir = './data/crop_39/train'

data_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

seed = 0

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

g = torch.Generator()
g.manual_seed(seed)

batch_size = 32
n_epochs = 500
holdout_sets = [[0], [1], [2], [4], [5], [6,7], [8]]

full_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)

array_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
holdout_set = holdout_sets[array_idx]

train_indices, val_indices = split_dataset(holdout_set, full_dataset)

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

lrs = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]

for learning_rate in lrs:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Linear(num_ftrs, 2)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
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
            
            pbar.set_postfix({'Set': holdout_set,
                              'LR': learning_rate,
                              'Epoch': epoch, 
                              'Validation Loss': f'{val_loss:.3f}', 
                              'Validation Accuracy': f'{val_acc:.3f}'})
    
    results_df = pd.DataFrame({
        'set': [holdout_set] * n_epochs,
        'learning_rate': [learning_rate] * n_epochs,
        'epoch': list(epochs),
        'epoch_loss': epoch_losses,
        'epoch_acc': epoch_accs
    })
    
    os.makedirs('./results/lr_tune', exist_ok=True)
    
    csv_path = f'./results/lr_tune/set-{holdout_set}_lr-{learning_rate}.csv'
    results_df.to_csv(csv_path, index=False)