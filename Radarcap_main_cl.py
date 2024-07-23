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
import pdb

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
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((368, 368)),
])

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


# 커스텀 데이터셋 생성
radar_range = [200, 1428]
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


image_data_path =  '/home/yunkwan/project/radarclip/data_test/image'
radar_data_path =  '/home/yunkwan/project/radarclip/data_test/radar'
sWVD_data_path ='/home/yunkwan/project/radarclip/data_test/swvd_folder'
paired_dataset_test = PairedDataset(image_root_dir=image_data_path, radar_root_dir=radar_data_path, radar_range=radar_range, transform=image_transform, radar_normalize=radar_normalize)
test_dataset = paired_dataset_test

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




class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.attention(query, key, value)
        return attn_output, attn_output_weights
embed_dim = 768  # 또는 images_features와 radars_features의 실제 차원으로 설정
num_heads = 8

# CrossAttention 모듈 초기화
cross_attention = CrossAttention(embed_dim, num_heads).to(device)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [seq_len, batch_size, embed_dim]
        x = x.mean(dim=0)  # [batch_size, embed_dim]
        return self.fc(x)
classifier = SimpleClassifier(embed_dim, 3).to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer,warmup_scheduler,cosine_scheduler,prepro_image_fn, num_epochs=100, early_stopping_patience=10000):
    best_val_loss = float('inf')
    patience_counter = 0
    
        # # load model
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)
    retrieval_model = AutoModel.from_pretrained("Yova/SmallCap7M")
    retrieval_model= retrieval_model.to(device)

    for param in retrieval_model.encoder.parameters():
        param.requires_grad = False
    
    # retrieval_model, feature_extractor_retrieval = clip.load("RN50x64", device=device)  #1024 size 448 
    
    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        cross_attention.train()
        # retrieval_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, radars,_,_,labels in train_loader_tqdm:
            images = images.to(device).float()
            # load from image_data_path
    
            radars = radars.to(device).float()
            labels = labels.to(device).long() 
        
            optimizer.zero_grad()
            
            radars_features = model(radars).reshape(-1,3)
            # pdb.set_trace()
            # radars_features = radars_features.unsqueeze(1)
            images_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)
            
            images_features = images_features.reshape(-1,50,768)[:,0:1,:]
            # images_features = images_features.unsqueeze(1)
            # pdb.set_trace()
            
            #cross attention

            # radars_features = radars_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
            # images_features = images_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
            # pdb.set_trace()
            # cross_attn_output, cross_attn_weights = cross_attention(radars_features, images_features, radars_features)
            # pdb.set_trace()
            # radars_features = classifier(cross_attn_output)
            # radars_features = model(images)

            loss = criterion(radars_features, labels)
            
            loss.backward()
            optimizer.step()
            
            #warmup 스케줄러 업데이트
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            

            running_loss += loss.item() * images.size(0)
            # 예측값 계산
            _, preds = torch.max(radars_features, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

            # train_loader_tqdm.set_postfix(loss=running_loss / len(train_dataset))
            train_loader_tqdm.set_postfix(loss=running_loss / total_predictions)

        # epoch_loss = running_loss / len(train_dataset)
        epoch_loss = running_loss / total_predictions
        epoch_acc = correct_predictions / total_predictions

 
        

        model.eval()
        classifier.eval()
        cross_attention.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0


        with torch.no_grad():
            for images, radars,_,_,labels in val_loader:
                images = images.to(device).float()
                radars = radars.to(device).float()
                labels = labels.to(device).long() 
            
              
                radars_features = model(radars).reshape(-1,3)
                
                # radars_features = radars_features.unsqueeze(1)
                images_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)
                images_features = images_features.reshape(-1,50,768)[:,0:1,:]
                # images_features = images_features.unsqueeze(1)
                # pdb.set_trace()
                # radars_features = radars_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
                # images_features = images_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
                # pdb.set_trace()
                # cross_attn_output, cross_attn_weights = cross_attention(radars_features, radars_features, radars_features)
                # pdb.set_trace()
                # radars_features = classifier(cross_attn_output)
            
                # radars_features = model(images)
                loss = criterion(radars_features, labels)        
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(radars_features, 1)
                correct_predictions += torch.sum(preds == labels).item()
                total_predictions += labels.size(0)

        val_acc = correct_predictions/total_predictions
        val_loss = val_loss / total_predictions
        
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train_acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f} , Val_acc: {val_acc:.4f}") 
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss, "train_accuracy": epoch_acc, "val_loss": val_loss, "val_accuracy": val_acc})


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
warmup_epochs = 20
base_lr = 1e-4
warmup_lr = 1e-5
# 워밍업 스케줄러 함수
def warmup_lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs) / base_lr
    else:
        return 1.0

def test_model(model, test_loader, criterion,prepro_image_fn):
    cross_attention.load_state_dict(torch.load('/home/yunkwan/project/radarclip/model_save/cross_attention'))
    classifier.load_state_dict(torch.load('/home/yunkwan/project/radarclip/model_save/classifier'))

    model.eval()
    cross_attention.eval()
    classifier.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
        # # load model
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)
    retrieval_model = AutoModel.from_pretrained("Yova/SmallCap7M")
    retrieval_model= retrieval_model.to(device)

    for param in retrieval_model.encoder.parameters():
        param.requires_grad = False

    with torch.no_grad():
        for images, radars,_,_,labels in test_loader:
            images = images.to(device).float()
            radars = radars.to(device).float()
            labels = labels.to(device).long()
            
            # radars_features = model(images)
            # radars_features = radars_features.unsqueeze(1)
            radars_features = model(radars).reshape(-1,3)
            images_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)
            images_features = images_features.reshape(-1,50,768)[:,0:1,:]
            # images_features = images_features.unsqueeze(1)
            
            # radars_features = radars_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
            # images_features = images_features.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
            # pdb.set_trace()
            # cross_attn_output, cross_attn_weights = cross_attention(radars_features, radars_features, radars_features)
            # pdb.set_trace()
            # radars_features = classifier(cross_attn_output)

     
            loss = criterion(radars_features, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(radars_features, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    test_loss = test_loss / total_predictions
    test_accuracy = correct_predictions/total_predictions
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Main function
def main(train_3, num_epochs, batch_size, test_mode):
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    import torchvision.models as models
    if test_mode:
        criterion = nn.CrossEntropyLoss()
        encoder_search_resnet1D_3 = ResNet1D_101(Bottleneck1D_101, [3, 4, 23, 3], 50*768).to(device)
        encoder_search_resnet1D_3.load_state_dict(torch.load("/home/yunkwan/project/radarclip/model_save/encoder_search_resnet1D_3.pth"))
        test_model(encoder_search_resnet1D_3, test_loader, criterion,prepro_image_fn= prepro_image_features_768)
        return
        
    encoder_search_resnet1D_3 = ResNet1D_101(Bottleneck1D_101, [3, 4, 23, 3], 3)
    # encoder_search_resnet1D_3 = models.resnet50(pretrained=False).to(device)
    # # 마지막 레이어를 수정하여 출력 크기를 1024로 변경
    # num_ftrs = encoder_search_resnet1D_3.fc.in_features
    # encoder_search_resnet1D_3.fc = nn.Linear(num_ftrs, 1024).to(device)


    # replace_bn_with_in(encoder_search_resnet1D_1024)
    encoder_search_resnet1D_3.to(device)
    
    import torchvision.models as models

    wandb.init(project="cross attention only")
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CosineSimilarity()
    # optimizer = optim.Adam(encoder_search_resnet1D_1024.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.AdamW(encoder_search_resnet1D_3.parameters(), lr=base_lr, weight_decay=1e-6)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    warmup_scheduler =  LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    
    
    
    
    train_model(encoder_search_resnet1D_3, train_loader, val_loader, criterion, optimizer,warmup_scheduler,cosine_scheduler, prepro_image_fn= prepro_image_features_768,num_epochs=num_epochs)
    save_model(encoder_search_resnet1D_3, "/home/yunkwan/project/radarclip/model_save/encoder_search_resnet1D_3.pth")
  
    save_model(cross_attention, '/home/yunkwan/project/radarclip/model_save/cross_attention')
    save_model(classifier, '/home/yunkwan/project/radarclip/model_save/classifier')
    test_model(encoder_search_resnet1D_3, test_loader, criterion,prepro_image_fn= prepro_image_features_768)
# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument('--train_3', action='store_true', help="Train the 1024 model")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--test_mode', action='store_true', help="Test mode")
    args = parser.parse_args()

    main(args.train_3, args.epochs, args.batch_size, args.test_mode)
