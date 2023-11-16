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
import optuna
import json
import plotly
import pickle

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

os.chdir('/home/kdoherty/spurge/data_release')

train_dir = './data/crop_39/train'

df = pd.read_csv('./results/best_lr.csv')
best_row = df.loc[df['accuracy'].idxmax()]
learning_rate = best_row['learning_rate']

def objective(trial):
    cutmix_prob = trial.suggest_float("cutmix_prob", 0.1, 1.0, step=0.1)
    cutmix_beta = trial.suggest_float("cutmix_beta", 0, 1.0, step=0.1)
    cutmix_num_mix = trial.suggest_int("cutmix_num_mix", 1, 3)

    stats = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    transform_list = [transforms.ToTensor(), stats]
    
    train_transforms = transforms.Compose(transform_list)
    val_transforms = transforms.Compose([transforms.ToTensor(), stats])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_dataset = CutMix(train_dataset, num_class=2, beta=cutmix_beta, prob=cutmix_prob, num_mix=cutmix_num_mix)
    
    val_dataset = datasets.ImageFolder(train_dir, transform=val_transforms)
    
    batch_size = 32
    n_epochs = 50

    seeds = range(8)

    seed_epoch_accs = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        # Shuffle and create subsets for training and validation
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:128]
        val_indices = indices[128:256]
    
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
        # Load pre-trained resnet50 model + higher level layers
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
        num_ftrs = model.fc.in_features
        
        model.fc = nn.Linear(num_ftrs, 2)
        
        criterion = CutMixCrossEntropyLoss(True).to(device)
    
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        epoch_accs = []
        
        with tqdm(total=n_epochs*len(train_loader), unit="batch", desc="Training Progress") as pbar:
            for epoch in range(n_epochs):
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
    
                train_loss = running_loss/len(train_loader)
    
                # Validate the model
                model.eval()
                running_loss = 0
                running_acc = 0
    
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device).long()
                    output = model(images)
                    output = output.unsqueeze(0) if len(output.shape) == 1 else output
                    loss = criterion(output, labels.unsqueeze(0) if labels.dim() == 0 else labels)
                    acc = binary_accuracy(output, labels.float())
                    running_loss += loss.item()
                    running_acc += acc.item()
    
                val_loss = running_loss/len(val_loader)
                val_acc = running_acc/len(val_loader)
                epoch_accs.append(val_acc)
                
                pbar.set_postfix({'Seed':seed,
                                  'Epoch': epoch+1, 
                                  'Validation Loss': f'{val_loss:.3f}', 
                                  'Validation Accuracy': f'{val_acc:.3f}'})
        
        seed_epoch_accs.append(epoch_accs)

    best_epoch, best_accuracy = best_cv_val(seed_epoch_accs)

    return best_accuracy

def save_study_cb(study, trial):
    # Directory where the figures and trial data will be saved
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    # Save figures
    if trial.state == optuna.trial.TrialState.COMPLETE:
        opt_hist_path = os.path.join(results_dir, f'cutmix_opt_history_plot.png')
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(opt_hist_path)

        opt_slice_path = os.path.join(results_dir, f'cutmix_opt_slice_plot.png')
        fig2 = optuna.visualization.plot_slice(study)
        fig2.write_image(opt_slice_path)
        
        if len(study.trials) > 1:
            opt_importance_path = os.path.join(results_dir, f'cutmix_opt_importance_plot.png')
            fig3 = optuna.visualization.plot_param_importances(study)
            fig3.write_image(opt_importance_path)

    study_path = os.path.join(results_dir, 'cutmix_opt_study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
trials = 500

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=trials, callbacks=[save_study_cb])

best_params_path = './results/best_cutmix.json'
best_params = study.best_params
with open(best_params_path, 'w') as f:
    json.dump(best_params, f)