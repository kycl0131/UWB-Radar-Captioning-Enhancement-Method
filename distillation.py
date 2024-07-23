import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from dataload import PairedDataset
from model import ResNet1D_101, Bottleneck1D_101
import wandb
from tqdm import tqdm
from transformers import AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import warnings
from transformers import ViTFeatureExtractor, AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from smallcap.src.vision_encoder_decoder import SmallCap, SmallCapConfig
from smallcap.src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from smallcap.src.utils import prep_strings, postprocess_preds
import json
from smallcap.src.retrieve_caps import *
import faiss
import torchvision.transforms as transforms
import pdb

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
wandb.require("core")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# class AdjustBrightnessContrast:
#     def __init__(self, alpha=1.0, beta=0):
#         self.alpha = alpha
#         self.beta = beta
    
#     def __call__(self, img):
#         img = np.array(img)
#         img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
#         img = Image.fromarray(img)
#         return img

# 이미지 데이터 전처리 및 변환 정의
image_transform = transforms.Compose([
    # AdjustBrightnessContrast(alpha=1, beta=0),
    # transforms.Resize((368, 368)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    
])

# 이미지 특징 추출 함수
def prepro_image_features(images, retrieval_model, device, batch_size=2, model_dim=768):
    resize_transform = transforms.Resize((224, 224))
    resized_images = torch.stack([resize_transform(image) for image in images])
    resized_images = resized_images.to(device).float()
    
    image_features_list = []
    for i in range(0, len(resized_images), batch_size):
        batch_images = resized_images[i:i + batch_size]
        torch.cuda.empty_cache()
        with torch.no_grad():
            # features = retrieval_model.encoder(batch_images.to(device))
            features = retrieval_model(batch_images.to(device))
            # image_features_batch = features.last_hidden_state.reshape(-1,50*model_dim)
            image_features_batch = features
        image_features_list.append(image_features_batch)
    image_features = torch.cat(image_features_list, axis=0)
    
    return image_features.to(device).float()

# 커스텀 데이터셋 생성
radar_range = [200, 1428]
radar_normalize = False
# 이미지 및 레이더 데이터셋 경로 설정
image_data_path =  '/home/yunkwan/project/radarclip/data_train/image'
radar_data_path =  '/home/yunkwan/project/radarclip/data_train/radar'
sWVD_data_path ='/home/yunkwan/project/radarclip/data_train/swvd_folder'
paired_dataset_train = PairedDataset(image_root_dir=image_data_path, radar_root_dir=radar_data_path, radar_range=radar_range, transform=image_transform, radar_normalize=radar_normalize)

train_dataset = paired_dataset_train

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
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import ViTModel, ViTConfig



class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.retrieval_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # self.classifier = nn.Linear(768, 3)
        self.resnet = ResNet1D_101(Bottleneck1D_101, [3, 4, 23, 3], 1*768)
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.reduce_dim = nn.Linear(768, 10)  # 크기 조정을 위한 레이어 추가
        self.fc = nn.Linear(10, 3)
    def forward(self, image, radar):
        image_features = self.retrieval_model(image).last_hidden_state[:, 0, :]
        # classfier = self.classifier(image_features)
        radar_features = self.resnet(radar).reshape(-1, 1, 768)
        image_features = image_features.reshape(-1, 1, 768)
        
        # pdb.set_trace()
        attention_output, _ = self.cross_attention(image_features.permute(1, 0, 2), radar_features.permute(1, 0, 2), radar_features.permute(1, 0, 2))
        attention_output = attention_output.mean(dim=0)
        reduced_features = self.reduce_dim(attention_output)  # 크기 조정
        output = self.fc(reduced_features)
        return output, reduced_features  # 조정된 크기의 특징 반환

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.resnet = ResNet1D_101(Bottleneck1D_101, [3, 4, 23, 3], 1*768)
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=16)
        self.fc1 = nn.Linear(768, 10)
        self.fc2 = nn.Linear(10, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, radar):
        radar_features = self.resnet(radar).reshape(-1,1,768)
        radar_features = radar_features.permute(1, 0, 2)
        
        attention_output, _ = self.attention(radar_features, radar_features, radar_features)
        attention_output = attention_output.permute(1, 0, 2).mean(dim=1)
        x = self.fc1(attention_output)
        x = self.dropout(x)
        output = self.fc2(x)
        return output, x

def distillation_loss(student_output, teacher_output, labels, alpha=0.5, temperature=3.0, feature_loss_weight=1.0):
    student_logits, student_features = student_output
    teacher_logits, teacher_features = teacher_output
    soft_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_logits / temperature, dim=1),
                                                    nn.functional.softmax(teacher_logits / temperature, dim=1)) * (alpha * temperature * temperature)
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels) * (1. - alpha)
    feature_loss = nn.MSELoss()(student_features, teacher_features) * feature_loss_weight
    return soft_loss + hard_loss + feature_loss

def train_model(teacher_model, student_model, train_loader, val_loader, criterion, optimizer, warmup_scheduler, cosine_scheduler, prepro_image_fn, num_epochs=100, early_stopping_patience=1000):
    best_val_loss = float('inf')
    patience_counter = 0

    # for param in retrieval_model.encoder.parameters():
    #     param.requires_grad = False

    for epoch in range(num_epochs):
        teacher_model.train()
        student_model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, radars, _, _, labels in train_loader_tqdm:
            torch.cuda.empty_cache()
            images = images.to(device).float()
            radars = radars.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()

            # image_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)
            # image_features = image_features #.reshape(-1, 50, 768)
            
            # 교사 모델 예측
            teacher_outputs, teacher_features   = teacher_model(images, radars)
            # teacher_vit_classifer loss 계산
            # teacher_vit_loss = criterion(teacher_vit_classifer, labels)*vit_loss
            
            # 학생 모델 예측
            student_outputs, student_features = student_model(radar=radars)

            # 손실 계산
            loss =  distillation_loss((student_outputs, student_features), (teacher_outputs, teacher_features), labels,
                                     alpha=wandb.config.alpha, temperature=wandb.config.temperature, feature_loss_weight=wandb.config.feature_loss_weight)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(student_outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

            train_loader_tqdm.set_postfix(loss=running_loss / total_predictions)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        epoch_loss = running_loss / total_predictions
        epoch_acc = correct_predictions / total_predictions

        teacher_model.eval()
        student_model.eval()
   
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            
            for images, radars, _, _, labels in val_loader:
                torch.cuda.empty_cache()
                images = images.to(device).float()
                radars = radars.to(device).float()
                labels = labels.to(device).long()

               
                # image_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)
                # image_features = image_features#.reshape(-1, 50, 768)
                
                # teacher_outputs, teacher_features, teacher_vit_classifer = teacher_model(images, radars)
                student_outputs, student_features = student_model(radar=radars)

                loss = criterion(student_outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(student_outputs, 1)
                correct_predictions += torch.sum(preds == labels).item()
                total_predictions += labels.size(0)

        val_acc = correct_predictions / total_predictions
        val_loss = val_loss / total_predictions
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train_acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val_acc: {val_acc:.4f}") 
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss, "train_accuracy": epoch_acc, "val_loss": val_loss, "val_accuracy": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    

def save_model(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def replace_bn_with_in(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            num_features = child.num_features
            eps = child.eps
            momentum = child.momentum
            affine = child.affine
            track_running_stats = child.track_running_stats
            in_layer = nn.InstanceNorm1d(num_features, eps, momentum, affine, track_running_stats)
            setattr(module, name, in_layer)
        else:
            replace_bn_with_in(child)

# 워밍업 스케줄러 설정
warmup_epochs = 20
base_lr = 1e-4
warmup_lr = 1e-5

def warmup_lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs) / base_lr
    else:
        return 1.0

def test_model(model, test_loader, criterion,prepro_image_fn):
    model.load_state_dict(torch.load("/home/yunkwan/project/radarclip/model_save/student_model.pth"))
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # for param in retrieval_model.encoder.parameters():
    #     param.requires_grad = False

    with torch.no_grad():
        for images, radars, _, _, labels in test_loader:
            torch.cuda.empty_cache()
            images = images.to(device).float()
            radars = radars.to(device).float()
            labels = labels.to(device).long()

            # image_features = prepro_image_fn(images, retrieval_model, device, batch_size=32)
            # image_features = image_features#.reshape(-1, 50, 768)
            
            student_outputs, student_features = model(radar=radars)

            loss = criterion(student_outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(student_outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    test_loss = test_loss / total_predictions
    test_accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
    wandb.finish()

def train():
    wandb.init()
    config = wandb.config

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,num_workers=4)
    
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)
    criterion = nn.CrossEntropyLoss()

    # AutoConfig.register("this_gpt2", ThisGPT2Config)
    # AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    # AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    # AutoConfig.register("smallcap", SmallCapConfig)
    # AutoModel.register(SmallCapConfig, SmallCap)
    # retrieval_model = AutoModel.from_pretrained("Yova/SmallCap7M").to(device)

    optimizer = optim.AdamW(
        list(teacher_model.parameters()) + 
        list(student_model.parameters())  ,
        lr=base_lr, weight_decay=1e-6
    )

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

    train_model(teacher_model, student_model, train_loader, val_loader, criterion, optimizer, warmup_scheduler, cosine_scheduler, prepro_image_fn=prepro_image_features, num_epochs=config.epochs)
    save_model(teacher_model, "/home/yunkwan/project/radarclip/model_save/teacher_model.pth")
    save_model(student_model, "/home/yunkwan/project/radarclip/model_save/student_model.pth")

    test_model(student_model, test_loader, criterion, prepro_image_fn=prepro_image_features)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--test_mode', action='store_true', help="Test mode")
    args = parser.parse_args()

    # Sweep 설정 정의
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'test_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'alpha': {
                'values': [0.3, 0.5, 0.7]
            },
            'temperature': {
                'values': [ 3.0, 4.0,5.0,6.0,7.0,8.0,9.0,10.0]
            },
            'feature_loss_weight': {
                'values': [0.1,0.2, 0.3,0.4,0.5,0,6, 0.7,0.8,0.9, 1.0]
            },
            'batch_size': {
                'values': [64]
            },
            'epochs': {
                'values': [30]
            
            }
        }
    }

    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project='teacher_student_distillation')

    # Sweep 에이전트 실행
    wandb.agent(sweep_id, train, count=40)
