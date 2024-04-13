# %%
import torch
from torchsummary import summary
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# %%
state_dict = torch.load('trained_model_sid_2.pth')

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
model=DeblurModel()

# %%
model.load_state_dict(state_dict)

# %%
new_img_path="\custom_test\generated" #output path where image we want to save the generated image

import os
img_directory = "mp2_test/custom_test/blur"

files = os.listdir(img_directory)

images_paths = [os.path.join(img_directory, file) for file in files]


# %%
# Plot one Image for visualization

try_img=cv2.imread(images_paths[30])
try_img = cv2.cvtColor(try_img, cv2.COLOR_BGR2RGB)

plt.imshow(try_img)
plt.show()

# %%
from PIL import Image

# Load the image using OpenCV
test_img = cv2.imread(images_paths[30])
test_img_pil = Image.fromarray(test_img)

preprocess = transforms.Compose([
    transforms.Resize((256, 448)),  # Resize the image
    transforms.ToTensor(),          # Convert to tensor
])
input_tensor = preprocess(test_img_pil).unsqueeze(0)  
model.eval()

with torch.no_grad():
    output_tensor = model(input_tensor)


# %%

# Assuming 'output_tensor' is your tensor obtained from the model
output_array = output_tensor.detach().cpu().numpy()  # Convert tensor to NumPy array
output_array = output_array.squeeze()
output_array = np.transpose(output_array, (1, 2, 0))  # Transpose to (height, width, channels)

# Rescale pixel values to valid range for image display (0 to 255)
output_array = ((output_array - output_array.min()) / (output_array.max() - output_array.min())) * 255

# Convert array to unsigned 8-bit integer (uint8) data type
output_array = output_array.astype(np.uint8)

# Display the image using OpenCV
cv2.imshow('Output Image', output_array)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# converting all blur image of test set into sharp image for comparison

for blur_img in images_paths:
    # print(blur_img)
    parts = blur_img.split("\\")
    # print(parts)
    output_path_3 = os.path.join(new_img_path, f"{parts[-1]}")
    # print(output_path_3)
    test_img = cv2.imread(blur_img)

    test_img_pil = Image.fromarray(test_img)
    
    preprocess = transforms.Compose([
        transforms.Resize((256, 448)),  
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(test_img_pil).unsqueeze(0) 
    
    model.eval()
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_array = output_tensor.detach().cpu().numpy()  
    output_array = output_array.squeeze()
    output_array = np.transpose(output_array, (1, 2, 0)) 
    
    # Rescale pixel values to valid range for image display (0 to 255)
    output_array = ((output_array - output_array.min()) / (output_array.max() - output_array.min())) * 255
    
    # Convert array to unsigned 8-bit integer (uint8) data type
    output_array = output_array.astype(np.uint8)

    
    cv2.imwrite(output_path_3, output_array)
    


