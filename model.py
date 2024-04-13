# %%
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image
from torchsummary import summary
from torch.cuda.amp import GradScaler, autocast

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_image_path = self.dataframe.iloc[idx, 0]
        final_image_path = self.dataframe.iloc[idx, 1]
        input_image = Image.open(input_image_path)
        final_image = Image.open(final_image_path)
        if self.transform:
            input_image = self.transform(input_image)
            final_image = self.transform(final_image)
        return input_image, final_image

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Dataset
dataframe = pd.read_csv("paths.csv")
dataframe = dataframe.sample(frac=1).reset_index(drop=True)



# %%
class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = DeblurModel().to(device)

# %%

from torch.utils.data import Subset, random_split

dataset = CustomDataset(dataframe, transform=transform) 
# train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create train and validation loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# %%
scaler = GradScaler()

# %%
num_epochs = 1

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

for epoch in range(num_epochs):
    total_train_loss = 0
    total_val_loss = 0
    
    # Training
    model.train()
    for i, (input_images, final_images) in enumerate(train_loader):
        input_images = input_images.to(device)
        final_images = final_images.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(input_images)
            train_loss = criterion(outputs, final_images)
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += train_loss.item()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader), train_loss.item()))
    
    # Validation
    model.eval()
    with torch.no_grad():
        for i, (input_images, final_images) in enumerate(val_loader):
            input_images = input_images.to(device)
            final_images = final_images.to(device)

            outputs = model(input_images)
            val_loss = criterion(outputs, final_images)
            total_val_loss += val_loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))

# %%
torch.save(model.state_dict(), 'trained_model_sid_2.pth')
print("Model saved successfully.")


