import os
import shutil
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from dataload import PairedDataset
from model import ResNet1D_101, Bottleneck1D_101
import wandb
from tqdm import tqdm
import clip
from sklearn.model_selection import train_test_split
import warnings
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR
# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")


# W&B 새로운 백엔드 사용
wandb.require("core")

# CUDA 및 CuDNN 설정 조정
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# # 폴더 경로 설정
image_folder = '/home/yunkwan/project/radarclip/data_train/image'
radar_folder = '/home/yunkwan/project/radarclip/data_train/radar'
backup_folder = '/home/yunkwan/project/radarclip/data_train/radar/backup'

# 백업 폴더가 없으면 생성
os.makedirs(backup_folder, exist_ok=True)

# 파일 이름에서 숫자만 추출하는 함수
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return match.group(0) if match else None

# 이미지 폴더와 레이더 폴더의 파일 이름 목록에서 숫자만 추출하여 가져오기
image_files = set(extract_number(f) for f in os.listdir(image_folder))
radar_files = os.listdir(radar_folder)

# 레이더 폴더의 파일 이름에서 숫자를 추출하여 이미지 폴더에 없는 경우 제거
for radar_file in radar_files:
    radar_file_number = extract_number(radar_file)
    if radar_file_number and radar_file_number not in image_files:
        radar_file_path = os.path.join(radar_folder, radar_file)
        backup_file_path = os.path.join(backup_folder, radar_file)
        os.remove(radar_file_path)
        print(f'Removed: {radar_file_path}')

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import cv2
class AdjustBrightnessContrast:
    def __init__(self, alpha=1.0, beta=0):
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, img):
        img = np.array(img)
        img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
        img = Image.fromarray(img)
        return img

# 이미지 및 레이더 데이터셋 경로 설정
image_data_path =  '/home/yunkwan/project/radarclip/data_train/image'
radar_data_path =  '/home/yunkwan/project/radarclip/data_train/radar'
sWVD_data_path ='/home/yunkwan/project/radarclip/data_train/swvd_folder'


# 이미지 데이터 전처리 및 변환 정의
image_transform = transforms.Compose([
    AdjustBrightnessContrast(alpha=1, beta=0),
    # transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((368, 368)),
])

# 커스텀 데이터셋 생성
radar_range = [200, 600]
radar_normalize = True
paired_dataset_train = PairedDataset(image_root_dir=image_data_path, radar_root_dir=radar_data_path, radar_range=radar_range, transform=image_transform, radar_normalize=radar_normalize)

train_dataset = paired_dataset_train
# train_dataloader = DataLoader(paired_dataset, batch_size=8, shuffle=True, num_workers=4)

# 이미지 및 레이더 Test 데이터셋 경로 설정
image_data_path =  '/home/yunkwan/project/radarclip/data_val/image'
radar_data_path =  '/home/yunkwan/project/radarclip/data_val/radar'
sWVD_data_path ='/home/yunkwan/project/radarclip/data_val/swvd_folder'
paired_dataset_val = PairedDataset(image_root_dir=image_data_path, radar_root_dir=radar_data_path, radar_range=radar_range, transform=image_transform, radar_normalize=radar_normalize)
val_dataset = paired_dataset_val
# val_dataloader = DataLoader(paired_dataset, batch_size=8, shuffle=True, num_workers=4)


# 데이터셋 인덱스 생성
# indices = list(range(len(paired_dataset)))
# train_indices, val_indices = train_test_split(indices, test_size=0.3, random_state=42)
# # val_indices, test_indices = train_test_split(val_indices, test_size=0.2, random_state=42)

# # Subset을 사용하여 훈련, 검증 및 테스트 데이터셋 생성
# train_dataset = Subset(paired_dataset, train_indices)
# val_dataset = Subset(paired_dataset, val_indices)
# # test_dataset = Subset(paired_dataset, test_indices)

# 특징 추출기 로드
global device
device = "cuda" if torch.cuda.is_available() else "cpu"


import torch
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig
from smallcap.src.vision_encoder_decoder import SmallCap, SmallCapConfig
from smallcap.src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from smallcap.src.utils import prep_strings, postprocess_preds
import json
from smallcap.src.retrieve_caps import *
import faiss
import torchvision.transforms as transforms



# 특징 추출기 로드



# 768 모델용 prepro_image_features 함수
def prepro_image_features_768(images, retrieval_model, device, batch_size=2):
    resize_transform = transforms.Resize((224, 224))
    resized_images = torch.stack([resize_transform(image) for image in images])
    resized_images = resized_images.to(device).float()#.half()
    
    image_features_list = []
    
    for i in range(0, len(resized_images), batch_size):
        batch_images = resized_images[i:i + batch_size]
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            features = retrieval_model.encoder(batch_images.to(device))
            image_features_batch = features.last_hidden_state.reshape(-1,50*768)

        image_features_list.append(image_features_batch)

    image_features = torch.concat(image_features_list, axis=0)
    
    return torch.Tensor(image_features).to(device).float()

# 1024 모델용 prepro_image_features 함수
def prepro_image_features_1024(images, retrieval_model, device, batch_size=2):
    """
    Resize images to (448, 448) and extract features using the retrieval model.
    
    Parameters:
    images (torch.Tensor): A batch of images with shape (batch_size, 3, 224, 224).
    retrieval_model: The model used to extract image features.
    device (str): The device to perform computations on ("cpu" or "cuda").
    batch_size (int): The number of images to process in each smaller batch.
    
    Returns:
    np.ndarray: The extracted image features.
    """
    # Define the transformation to resize the images
    resize_transform = transforms.Resize((448, 448))
    
    # Apply the transformation to each image in the batch
    resized_images = torch.stack([resize_transform(image) for image in images])
    
    # Move the resized images to the device and use mixed precision
    resized_images = resized_images.to(device).float() #.half()
    
    # Process images in smaller batches to avoid memory overflow
    image_features_list = []
    
    for i in range(0, len(resized_images), batch_size):
        batch_images = resized_images[i:i + batch_size]
        # Clear cache to optimize memory
        torch.cuda.empty_cache()
        # 이미지 특징 추출
        with torch.no_grad():
            image_features_batch = retrieval_model.encode_image(batch_images) #.cpu().detach().numpy()
            # print(image_features_batch.shape)
        image_features_list.append(image_features_batch)
    
    # Combine all the batch features
    image_features = torch.concat(image_features_list, axis=0)
    
    return torch.Tensor(image_features).to(device).float()


fs = 23.328e9  # 23.328e9  샘플링 주파수 예시
from scipy.signal import butter, filtfilt
# 밴드패스 필터 설계
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 필터 적용
lowcut = 6.5e9  # 6.5 GHz
highcut = 10e9  # 10 GHz


def train_model(model, retrieval_model, train_loader, val_loader, criterion, optimizer,warmup_scheduler,cosine_scheduler, prepro_image_fn, num_epochs=100, early_stopping_patience=10000, filter= None):
    best_val_loss = float('inf')
    patience_counter = 0
    
        
    for epoch in range(num_epochs):
        model.train()
        filter.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, radars,_,_ in train_loader_tqdm:
            images = images.to(device).float()
            radars = radars.to(device).float()
                    
            # 배치를 개별 이미지로 분리
            # b_size, height, width = swvds.shape
            # rgb_images = []

            # for i in range(b_size):
            #     # 이미지 데이터 numpy 배열로 변환
            #     img_np = swvds[i].numpy()
                
            #     # 데이터 정규화 (필요시)
            #     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
            #     # 컬러맵을 적용하여 RGB 이미지로 변환
            #     img_rgb = plt.cm.viridis(img_np)[:, :, :3]  # (height, width, 3)
                
            #     # RGB 이미지를 텐서로 변환하고 배치 차원을 추가
            #     img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)  # (3, height, width)
            #     rgb_images.append(img_tensor)

            # # RGB 이미지들을 하나의 배치 텐서로 결합
            # rgb_images_batch = torch.stack(rgb_images)  # (batch_size, 3, height, width)
            # rgb_images_batch = rgb_images_batch.to(device).float()

            if filter is not None:
                radars = filter(radars).to(device).float()
            optimizer.zero_grad()

            radars_features = model(radars)
            # radars_features = model(rgb_images_batch)
            images_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)
            
            loss = criterion(radars_features, images_features).mean()
            # loss = 1- criterion(radars_features, images_features).mean()
            
            loss.backward()
            optimizer.step()
            
            #warmup 스케줄러 업데이트
            if epoch < 5:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            

            running_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_dataset))

        epoch_loss = running_loss / len(train_dataset)
        wandb.log({"train_loss": epoch_loss, "epoch": epoch + 1})

        model.eval()
        filter.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, radars,_,_ in val_loader:
                images = images.to(device).float()
                radars = radars.to(device).float()
                #        # 배치를 개별 이미지로 분리
                # b_size, height, width = swvds.shape
                # rgb_images = []

                # for i in range(b_size):
                #     # 이미지 데이터 numpy 배열로 변환
                #     img_np = swvds[i].numpy()
                    
                #     # 데이터 정규화 (필요시)
                #     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    
                #     # 컬러맵을 적용하여 RGB 이미지로 변환
                #     img_rgb = plt.cm.viridis(img_np)[:, :, :3]  # (height, width, 3)
                    
                #     # RGB 이미지를 텐서로 변환하고 배치 차원을 추가
                #     img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)  # (3, height, width)
                #     rgb_images.append(img_tensor)

                # # RGB 이미지들을 하나의 배치 텐서로 결합
                # rgb_images_batch = torch.stack(rgb_images)  # (batch_size, 3, height, width)
                # rgb_images_batch = rgb_images_batch.to(device).float()
            
                if filter is not None:
                    radars = filter(radars).to(device).float()

                radars_features = model(radars)
                # radars_features = model(rgb_images_batch)
                images_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)

                loss = criterion(radars_features, images_features).mean()
                # loss = 1- criterion(radars_features, images_features).mean()
                
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_dataset)
        wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    wandb.finish()

# 모델 저장 함수
def save_model(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



class LearnableLogFilter(nn.Module):
    def __init__(self, input_shape, device):
        super(LearnableLogFilter, self).__init__()
        self.log_scale = nn.Parameter(torch.ones(1, *input_shape[1:], device=device))
        self.log_shift = nn.Parameter(torch.zeros(1, *input_shape[1:], device=device))
        # self.initialized = True  # 초기화 상태를 true로 설정

    def forward(self, x):
        # x = x.cpu().detach().numpy()
        # x = bandpass_filter(x, lowcut, highcut, fs)
        # x = torch.Tensor(x.copy()).to(device)
        # Ensure positive input for log
        # x =x * self.log_scale + self.log_shift 
        # x = torch.clamp(x, min=1e-9)
        return x #* self.log_scale + self.log_shift #torch.log(x * self.log_scale + self.log_shift)
    
class ResNet1DWithFilter(nn.Module):
    def __init__(self, resnet_model,pre_filter):
        super(ResNet1DWithFilter, self).__init__()
        self.log_filter = pre_filter
        self.resnet_model = resnet_model

    def forward(self, x):
        x = self.log_filter(x)
        x = self.resnet_model(x)
        return x
    
def replace_bn_with_in(module):
    """
    Recursively replace all BatchNorm layers with InstanceNorm layers in the given module.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            # Get the parameters of the original BatchNorm layer
            num_features = child.num_features
            eps = child.eps
            momentum = child.momentum
            affine = child.affine
            track_running_stats = child.track_running_stats
            
            # Create a new InstanceNorm layer with the same parameters
            in_layer = nn.InstanceNorm1d(num_features, eps, momentum, affine, track_running_stats)
            
            # Replace the BatchNorm layer with the InstanceNorm layer
            setattr(module, name, in_layer)
        else:
            # Recursively apply to child layers
            replace_bn_with_in(child)


# 워밍업 스케줄러 설정
warmup_epochs = 10
base_lr =10e-5# 0.001
warmup_lr =10e-6 # 0.0001
# 워밍업 스케줄러 함수
def warmup_lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs) / base_lr
    else:
        return 1.0

# Main function
def main(train_1024, train_768, num_epochs, batch_size):
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if train_1024:
        encoder_search_resnet1D_1024 = ResNet1D_101(Bottleneck1D_101, [3, 4, 23, 3], 1024)
        # replace_bn_with_in(encoder_search_resnet1D_1024)
        encoder_search_resnet1D_1024.to(device)
        
        import torchvision.models as models

        # # 사전 훈련된 ResNet-50 모델 로드
        # encoder_search_resnet1D_1024 = models.resnet50(pretrained=False).to(device)
        # # 마지막 레이어를 수정하여 출력 크기를 1024로 변경
        # num_ftrs = encoder_search_resnet1D_1024.fc.in_features
        # encoder_search_resnet1D_1024.fc = nn.Linear(num_ftrs, 1024).to(device)


        LearnableLogFilter_1024 = LearnableLogFilter((1,1,radar_range[1]-radar_range[0]),device)
        
        
        wandb.init(project="encoder_search_resnet1D_1024")
        criterion = nn.MSELoss()
        # criterion = nn.CosineSimilarity()
        # optimizer = optim.Adam(encoder_search_resnet1D_1024.parameters(), lr=0.001, weight_decay=1e-5)
        optimizer = optim.AdamW(encoder_search_resnet1D_1024.parameters(), lr=base_lr, weight_decay=1e-5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        warmup_scheduler =  LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
        
        retrieval_model, feature_extractor_retrieval = clip.load("RN50x64", device=device)  #1024 size 448 
        for param in retrieval_model.parameters():
            param.requires_grad = False
        train_model(encoder_search_resnet1D_1024, retrieval_model, train_loader, val_loader, criterion, optimizer,warmup_scheduler,cosine_scheduler, prepro_image_features_1024, num_epochs=num_epochs,filter = LearnableLogFilter_1024)
        save_model(encoder_search_resnet1D_1024, "/home/yunkwan/project/radarclip/model_save/encoder_search_resnet1D_1024.pth")
        save_model(LearnableLogFilter_1024,'/home/yunkwan/project/radarclip/model_save/LearnableLogFilter_1024')

    if train_768:
        import torchvision.models as models
        encoder_search_resnet1D_768 = ResNet1D_101(Bottleneck1D_101, [3, 4, 23, 3], 768*50)
        # replace_bn_with_in(encoder_search_resnet1D_768)
        encoder_search_resnet1D_768.to(device)
        # 사전 훈련된 ResNet-50 모델 로드
        # encoder_search_resnet1D_768 = models.resnet50(pretrained=False).to(device)
        # # 마지막 레이어를 수정하여 출력 크기를 1024로 변경
        # num_ftrs = encoder_search_resnet1D_768.fc.in_features
        # encoder_search_resnet1D_768.fc = nn.Linear(num_ftrs, 768*50).to(device)




        LearnableLogFilter_768 =   LearnableLogFilter((1,1,radar_range[1]-radar_range[0]),device)
        
        wandb.init(project="encoder_search_resnet1D_768")
        criterion = nn.MSELoss()
        # criterion = nn.CosineSimilarity()
        # optimizer = optim.Adam(encoder_search_resnet1D_768.parameters(), lr=0.001, weight_decay=1e-5)
        optimizer = optim.AdamW(encoder_search_resnet1D_768.parameters(), lr=base_lr, weight_decay=1e-5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)


        
        
        # # load model
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoConfig.register("smallcap", SmallCapConfig)
        AutoModel.register(SmallCapConfig, SmallCap)
        model = AutoModel.from_pretrained("Yova/SmallCap7M")
        model= model.to(device)
        retrieval_model = model
        # from transformers import CLIPFeatureExtractor, CLIPVisionModel
        # encoder_name = 'openai/clip-vit-base-patch32'
        # feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name) 
        # clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)
        # retrieval_model = clip_encoder

        for param in retrieval_model.encoder.parameters():
            param.requires_grad = False
        train_model(encoder_search_resnet1D_768, retrieval_model, train_loader, val_loader, criterion, optimizer,warmup_scheduler,cosine_scheduler, prepro_image_features_768, num_epochs=num_epochs,filter=LearnableLogFilter_768)
        save_model(encoder_search_resnet1D_768, "/home/yunkwan/project/radarclip/model_save/encoder_search_resnet1D_768.pth")
        save_model(LearnableLogFilter_768, '/home/yunkwan/project/radarclip/model_save/LearnableLogFilter_768')


# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument('--train_1024', action='store_true', help="Train the 1024 model")
    parser.add_argument('--train_768', action='store_true', help="Train the 768 model")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    args = parser.parse_args()

    main(args.train_1024, args.train_768, args.epochs, args.batch_size)
