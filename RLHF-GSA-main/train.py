import torch
from tqdm import tqdm
import os
import sys
from models.model import MatchingModel
sys.path.append(os.path.abspath(os.path.dirname('__file__')))
from models.process import create_matching_dataloader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch import nn 
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(data_path, prev_epoch,batch_size, learning_rate, epoch_nums, weights_path):
    full_loader = create_matching_dataloader(data_path, batch_size=1, shuffle=False)
    dataset = full_loader.dataset
    total_size = len(dataset)
    train_size = int(total_size * 0.9)
    val_size = total_size - train_size

    train_set = Subset(dataset, list(range(train_size)))
    val_set = Subset(dataset, list(range(train_size, total_size)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print('model initialization...')

    model = MatchingModel(). to (device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.98, 0.9))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    if prev_epoch != 0:
        checkpoint_path = os.path.join(weights_path, f"{prev_epoch}_statedict.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
    total_params = sum(p.numel() for p in tqdm(model.parameters(), desc='caculating trainable parameters...') if p.requires_grad)
    print(f'###### total trainable parameter: {total_params}({(total_params/1000000000):.3f}B) ######')
    
    for epoch in range(epoch_nums):
        model.train()
        epoch_train_loss = 0 

        for userA, userB, label in tqdm(train_loader, desc=f"[{epoch+1}/{epoch_nums}] Training"):
            userA = {k: v.to(device) for k, v in userA.items()}
            userB = {k: v.to(device) for k, v in userB.items()}
            label = label.to(device)

            pred_prob = model(userA, userB).squeeze(1)  

            weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
            label_weighted = torch.sigmoid((label[:, :4] * weights).sum(dim=1))  

            loss = loss_fn(pred_prob, label_weighted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss  = 0
        with torch.no_grad():
            for userA, userB, label in tqdm(val_loader, desc=f"[{epoch+1}/{epoch_nums}] Validating"):
                userA = {k: v.to(device) for k, v in userA.items()}
                userB = {k: v.to(device) for k, v in userB.items()}
                label = label.to(device)

                pred_prob = model(userA, userB).squeeze(1)
                label_weighted = torch.sigmoid((label[:, :4] * weights).sum(dim=1))

                loss = loss_fn(pred_prob, label_weighted)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(weights_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(weights_path, f"best_model_epoch{epoch+1}.pth"))
            print(f"Best model saved at epoch {epoch+1} with total val loss {best_val_loss:.4f}")


    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    dims = ["Gender", "School", "College", "MBTI"]

    for i in range(4):
        plt.figure()
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f"{dims[i]} Similarity Loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"loss_curve_{dims[i].lower()}.png")

    print("Loss curves saved.")

if __name__ == "__main__":
    train_model(data_path="./mentee.xlsx", prev_epoch=0, batch_size=16, learning_rate=1e-4, epoch_nums=50, weights_path="./weights")
