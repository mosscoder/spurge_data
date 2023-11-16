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

device = torch.device("cuda")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class RandomGaussianBlur:
    def __init__(self, kernel_size=3, probability=0.5):
        self.kernel_size = kernel_size
        self.probability = probability
        self.gaussian_blur = transforms.GaussianBlur(self.kernel_size)

    def __call__(self, img):
        if random.random() < self.probability:
            return self.gaussian_blur(img)
        return img

def binary_accuracy(preds, y):
    # preds are the raw scores output by the model
    # Convert these scores to predicted class indices
    preds_class = preds.argmax(dim=1)
    # Compare the predicted classes with the true classes
    correct = (preds_class == y).float()
    # Calculate the accuracy
    acc = correct.sum() / len(correct)
    return acc
    
# Modify the data loaders to initialize each worker with the same seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

os.chdir('/home/kdoherty/spurge/data_release')

train_dir = './data/crop_39/train'

seed = 0
batch_size = 32

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

g = torch.Generator()
g.manual_seed(seed)

df = pd.read_csv('./results/best_lr.csv')
best_row = df.loc[df['accuracy'].idxmax()]
learning_rate = best_row['learning_rate']
print(f'Learning rate: {learning_rate}')

with open('./results/best_augs.json', 'r') as file:
    augs = json.load(file)

with open('./results/best_epoch.json', 'r') as file:
    n_epochs = json.load(file)['n_epochs']
print(f'Epoch count: {n_epochs}')

gaussian_blur = augs['gaussian_blur']
flip_horizontal = augs['flip_horizontal']
flip_vertical = augs['flip_vertical']
brightness = augs['brightness']
contrast = augs['contrast']
saturation = augs['saturation']
hue = augs['hue']
rotation = augs['rotation']

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

print(train_transforms)

full_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

# Create the data loaders
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

# Load pre-trained resnet50 model + higher level layers
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss().to(device)

model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

epochs = range(n_epochs)

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
        
        pbar.set_postfix({'Epoch': epoch, 
                          'Training Loss': f'{train_loss:.3f}',
                         })

test_dir = './data/crop_39/val'

# Apply the same normalization stats but not the augmentation transformations for the test set
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    stats
])

# Load the test data
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Create the test DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to evaluate the model on the test set
def evaluate(model, loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_acc = 0
    with torch.no_grad():  # No gradients required for evaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            output = model(images)
            output = output.unsqueeze(0) if len(output.shape) == 1 else output
            loss = criterion(output, labels.unsqueeze(0) if labels.dim() == 0 else labels)
            test_loss += loss.item()
            acc = binary_accuracy(output, labels)
            test_acc += acc.item()
    
    test_loss /= len(loader)
    test_acc /= len(loader)
    return test_loss, test_acc

# Load the model weights if necessary, for example:
# model.load_state_dict(torch.load('path_to_your_saved_model.pth'))

# Calculate loss and accuracy on the test set
test_loss, test_accuracy = evaluate(model, test_loader)

print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

test_perf_dir = './results/test_performance'
os.makedirs(test_perf_dir, exist_ok=True)
result_path = os.path.join(test_perf_dir, 'baseline.json')

with open(result_path, 'w') as f:
    json.dump({'loss':test_loss,'accuracy':test_accuracy}, f)