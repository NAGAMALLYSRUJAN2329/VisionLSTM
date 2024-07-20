import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import h5py
from glob import glob
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import math

from vision_lstm_sru import VisionLSTM

def get_data(data_path):
    d=np.zeros((len(data_path), 1, 128, 128))
    for i, path in enumerate(data_path):
            with h5py.File(path, 'r') as hdf:
                key=list(hdf.keys())[0]
                data = np.array(hdf.get(key))
                if key=='mask':
                    data=data.reshape(128,128,1)
                temp=data[:, :, 0]
                d[i, 0, :, :] = temp
    return d

TRAIN_PATH = r"data/TrainData/img/*.h5"
TRAIN_MASK = r"data/TrainData/mask/*.h5"
# VAL_PATH = r"data/ValidData/img/*.h5"
# TEST_PATH = r"data/TestData/img/*.h5"
all_train = sorted(glob(TRAIN_PATH))
all_mask = sorted(glob(TRAIN_MASK))
# all_val = sorted(glob(VAL_PATH))
# all_test = sorted(glob(TEST_PATH))

train_data=get_data(all_train)
mask_data=get_data(all_mask)
# val_data=get_data(all_val)
# test_data=get_data(all_test)
print("data loaded",train_data.shape, mask_data.shape)

l=int(train_data.shape[0]*0.8)
val_data=train_data[l:,:,:,:]
val_mask_data=mask_data[l:,:,:,:]

train_data=train_data[:l,:,:,:]
mask_data=mask_data[:l,:,:,:]
print("data shape(train, mask, val, val mask)",train_data.shape, mask_data.shape, val_data.shape, val_mask_data.shape)

class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image).permute(1, 2, 0)
            image=(image-image.min())/(image.max()-image.min())
            mask = self.transform(mask).permute(1, 2, 0)
        return image, mask

def load_data(train_data, mask_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # transform=None
    dataset = CustomDataset(train_data, mask_data, transform=transform)
    return DataLoader(dataset, batch_size=8, shuffle=True)

class PrintLayer(nn.Module):
    def __init__(self, layer_name):
        super(PrintLayer, self).__init__()
        self.layer_name = layer_name
    def forward(self, x):
        if torch.isnan(x).any():
            print(f'nan values found in output layer of {self.layer_name}')
            # raise ValueError(f'nan valur found in layer {self.layer_name}')
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)

        self.vlm = VisionLSTM(
            dim=256,
            depth=24,
            patch_size=16,
            input_shape=(1, 1, 1),
            output_shape=None,
            drop_path_rate=0.05,
            stride=None,
            mode=None,
            pooling=None,
        )

        self.bottleneck = self.conv_block(128, 256)

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(32, 16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

        # self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            # PrintLayer('Before_Conv2d_1'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # PrintLayer('Conv2d_1'),
            nn.BatchNorm2d(out_channels),
            # PrintLayer('BatchNorm2d_1'),
            nn.ReLU(inplace=True),
            # PrintLayer('ReLU_1'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # PrintLayer('Conv2d_2'),
            nn.BatchNorm2d(out_channels),
            # PrintLayer('BatchNorm2d_2'),
            nn.ReLU(inplace=True),
            # PrintLayer('ReLU_2'),
        )

    def forward(self, x):
        # if torch.isnan(x).any():
        #     raise ValueError("Contain nan values")
        enc1 = self.encoder1(x)
        # if torch.isnan(enc1).any():
            # print(enc1)
            # raise ValueError("Contain nan values")
        enc2 = self.encoder2(self.downsample(enc1))
        # if torch.isnan(enc2).any():
        #     raise ValueError("Contain nan values")
        enc3 = self.encoder3(self.downsample(enc2))
        # if torch.isnan(enc3).any():
        #     raise ValueError("Contain nan values")
        enc4 = self.encoder4(self.downsample(enc3)) # [B, 128, 16, 16]

        # if torch.isnan(enc4).any():
        #     raise ValueError("Contain nan values before vlm")
        enc4=enc4.reshape(enc4.shape[0], enc4.shape[1], -1)
        enc4=self.vlm(enc4)
        n = int(math.sqrt(enc4.shape[2]))
        enc4=enc4.reshape(enc4.shape[0], enc4.shape[1], n, n)
        # print(enc4.shape)
        # if torch.isnan(enc4).any():
        #     raise ValueError("Contain nan values after vlm")

        bottleneck = self.bottleneck(self.downsample(enc4))
        # print(bottleneck.shape)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

    def downsample(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)


def check_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                raise RuntimeError(f'NaNs in gradients for parameter {name}')

def check_nan_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise RuntimeError(f'NaNs in parameter {name}')
        
model = UNet(in_channels=1, out_channels=1)
torch.autograd.set_detect_anomaly(True)

print("model output shape:",model(torch.randn(1,1,128,128)).shape)

learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)
model = UNet(in_channels=1, out_channels=1).float().to(device)
model=torch.compile(model)
torch.set_float32_matmul_precision('high')
# model = torch.jit.script(model)
loss_fn = nn.BCELoss()
# loss_fn = dice_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
train_loader = load_data(train_data, mask_data)
valid_loader = load_data(val_data, val_mask_data)
val_needed = True

train_loss_history = []
val_loss_history = []
train_precision_history = []
val_precision_history = []
train_recall_history = []
val_recall_history = []
train_f1_history = []
val_f1_history = []

import tqdm
import time
num_epochs = 100
highest_f1=0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_f1=0
    train_precision=0
    train_recall=0
    skipped_count=0
    # time_1=time.time()
    for i, (images, masks) in enumerate(train_loader):
        print(i,epoch)
        # print('1',time.time()-time_1)
        # time_1=time.time()

        # for i, (name, param) in enumerate(model.named_parameters()):
        #     if i==0:
        #         print(f"Layer: {name} | Size: {param.size()} | Values:")
        #         print(param)
        #         print()

        images = images.float().to(device)
        masks = masks.float().to(device)
        # if torch.isnan(images).any() or torch.isnan(masks).any():
        #     if torch.isnan(images).any():
        #         print("images ", i)
        #         print(images)
        #     if torch.isnan(masks).any():
        #         print("masks ", i)
        #         print(masks)
            # raise ValueError("Input contains NaNs!")
        # if torch.isinf(images).any() or torch.isinf(masks).any():
        #     raise ValueError("Input contains Infs!")
        optimizer.zero_grad()
        # print('2',time.time()-time_1)
        # time_1=time.time()
        outputs = model(images)
        # print('3',time.time()-time_1)
        # time_1=time.time()
        # if outputs.shape != masks.shape:
        #     raise ValueError(f"Output shape {outputs.shape} does not match target shape {masks.shape}")
        # print(outputs)
        if not torch.any(torch.isnan(outputs)):
            loss = loss_fn(outputs, masks)
            loss.backward()
            # print('4',time.time()-time_1)
            # time_1=time.time()
            # print(f"Gradients in epoch {epoch + 1}, batch {i + 1}:")
            # for i, (name, param) in enumerate(model.named_parameters()):
            #     if param.grad is not None and i==0:
            #         print(f"Layer: {name} | Size: {param.size()} | Gradient:")
            #         print(param.grad)
            #         print()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            # check_nan_gradients(model)
            # check_nan_parameters(model)
            optimizer.step()
            # print('5',time.time()-time_1)
            time_1=time.time()
            train_loss += loss.item()
            preds = (outputs > 0.2).float()
            masks=masks.cpu().numpy().flatten()
            preds=preds.cpu().numpy().flatten()
            train_f1 += f1_score(masks, preds)
            train_precision += precision_score(masks, preds, zero_division=1)
            train_recall += recall_score(masks, preds, zero_division=1)
            # print('6',time.time()-time_1)
            # time_1=time.time()
        else:
            # print('_'*100)
            # print(outputs)
            skipped_count+=1
            # loss = loss_fn(outputs, masks)
            print(f"total Skipped batches: {skipped_count} in epoch: {epoch+1}, skipped batch number: {i+1}")
            raise ValueError(f"total Skipped batches: {skipped_count} in epoch: {epoch+1}, skipped batch number: {i+1}")
    train_loss /= len(train_loader)
    train_f1 /= len(train_loader)
    train_precision /= len(train_loader)
    train_recall /= len(train_loader)

    val_loss = train_loss
    val_f1 = train_f1
    val_precision = train_precision
    val_recall = train_recall
    if val_needed:
        model.eval()

        val_loss = 0
        val_f1 = 0
        val_precision = 0
        val_recall = 0

        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.float().to(device)
                masks = masks.float().to(device)

                outputs = model(images)
                if torch.isnan(outputs).any():
                    continue
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                preds = outputs > 0.5
                masks=masks.cpu().numpy().flatten()
                preds=preds.cpu().numpy().flatten()
                val_f1 += f1_score(masks, preds)
                val_precision += precision_score(masks, preds, zero_division=1)
                val_recall += recall_score(masks, preds, zero_division=1)
            val_loss /= len(valid_loader)
            val_f1 /= len(valid_loader)
            val_precision /= len(valid_loader)
            val_recall /= len(valid_loader)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_precision_history.append(train_precision)
    val_precision_history.append(val_precision)
    train_recall_history.append(train_recall)
    val_recall_history.append(val_recall)
    train_f1_history.append(train_f1)
    val_f1_history.append(val_f1)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f},Train F1: {train_f1:.4f},Train Precision: {train_precision:.4f},Train Recall: {train_recall:.4f}')
    if val_needed:
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')

    if val_f1 > highest_f1:
        highest_f1 = val_f1
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"Saved the model to best_model.pth with validation f1 of {val_f1}")
    torch.save(model.state_dict(), "models/latest_model.pth")
print("Training complete")