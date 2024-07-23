from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import joblib


class PairedDataset(Dataset):
    def __init__(self, image_root_dir, radar_root_dir, sWVD_root_dir=None, radar_range=[0, 1488], transform=None, radar_normalize=False,PCA_mode=False,n_components= 2):
        self.image_root_dir = image_root_dir
        self.radar_root_dir = radar_root_dir
        self.sWVD_root_dir = sWVD_root_dir
        self.start = radar_range[0]
        self.end = radar_range[1]
        self.transform = transform
        self.radar_normalize = radar_normalize
        self.PCA_mode = PCA_mode
        self.classes = sorted(os.listdir(image_root_dir))
        self.pairs = []
        self.missing_sWVD = []
        self.all_cls_pairs = []
        for cls in self.classes:
            image_dir = os.path.join(self.image_root_dir, cls)
            radar_dir = os.path.join(self.radar_root_dir, cls)
            sWVD_dir = os.path.join(self.sWVD_root_dir, cls) if self.sWVD_root_dir else None
            
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('png')])
            radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('npy')])
            if self.sWVD_root_dir is not None:
                sWVD_files = sorted([f for f in os.listdir(sWVD_dir) if f.endswith('npy')])
                sWVD_ids = {os.path.splitext(f)[0].split('_')[-1]: os.path.join(sWVD_dir, f) for f in sWVD_files}
                
            # 번호를 추출하여 매칭
            image_ids = {os.path.splitext(f)[0].split('_')[-1]: os.path.join(image_dir, f) for f in image_files}
            radar_ids = {os.path.splitext(f)[0].split('_')[-1]: os.path.join(radar_dir, f) for f in radar_files}
            # according to cls  -> label
            label = {'lay': 0, 'sit': 1, 'stand': 2}


            cls_pairs = [(image_ids[key], radar_ids[key], label.get(cls, None), key) for key in image_ids if key in radar_ids]
            self.all_cls_pairs.extend(cls_pairs)
        
            
        for img, radar,label, key in self.all_cls_pairs:
            if self.sWVD_root_dir is not None:
                sWVD_path = sWVD_ids.get(key, None)
                # check sWVD_path exist the file
                if not os.path.exists(sWVD_path):
                    print(f"No sWVD file found for {radar}")
                    self.missing_sWVD.append((img, radar))
                    continue
                self.pairs.append((img, radar, sWVD_path))
                # print(f"Matched pairs with sWVD: {self.pairs}")
            else:
                self.pairs.append((img, radar,label))
            
        # 디버깅을 위해 sWVD 매칭 출력 
        if not self.pairs:
            raise ValueError("No matching image and radar files found.")
        
        if self.missing_sWVD:
            raise ValueError(f"No matching sWVD files found for the following radar files: {self.missing_sWVD}")

        print(f"Found {len(self.pairs)} pairs of images and radar data.")
        # print(self.pairs)

        # 전체 레이더 데이터로 PCA 학습
        self.n_components = n_components  # You can set the desired number of PCA components
        radar_range_str = "_".join(map(str, radar_range))
        self.pca_model_path = f"/home/yunkwan/project/radarclip/model_save/pca_model_{self.n_components}_{radar_range_str}.pkl"

        if self.PCA_mode == True:
            if os.path.exists(self.pca_model_path):
                self.pca = joblib.load(self.pca_model_path)
                print(f"PCA model loaded from {self.pca_model_path}")
            else:
                self.pca = PCA(n_components=self.n_components)
                self.fit_pca()
                joblib.dump(self.pca, self.pca_model_path)
                print(f"PCA model saved to {self.pca_model_path}")
            
    def fit_pca(self):
        radar_data_all = []
        for _, radar_path, _ in self.pairs:
            radar_data = np.load(radar_path)[self.start:self.end]
            radar_data_all.append(radar_data.flatten())
        radar_data_all = np.array(radar_data_all)
        self.pca.fit(radar_data_all)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image_path, radar_path, label = pair[:3]
    
        if self.sWVD_root_dir is not None:
            sWVD_path = pair[2] if len(pair) > 2 else None
           
        
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 레이더 데이터 로드
        radar = np.load(radar_path)
        
        radar = radar[0, self.start:self.end] #-radar[1, self.start:self.end]
        
        radar = radar.reshape(1, -1)
        
        if self.PCA_mode == True:
            radar = self.pca.transform(radar)
        radar = torch.tensor(radar, dtype=torch.float32)
        radar = radar.reshape(1, -1)
        # index 200 이상에서 기존값에 100을 곱합
        radar = radar


        
        # 레이더 데이터 노멀라이즈 -1 to 1
        if self.radar_normalize:
            # radar_max_min = torch.Tensor([radar.max(dim=1, keepdim=True)[0], radar.min(dim=1, keepdim=True)[0]]).reshape(1, 2)
            min_vals = radar.min(dim=1, keepdim=True)[0]
            max_vals = radar.max(dim=1, keepdim=True)[0]
            normalized_radar =  2*(radar - min_vals) / (max_vals - min_vals) -1

            # normalized_radar = (radar+0.0003549871524764846)/ 0.14883218200478288
            # normalized_radar = 100 * (radar - min_vals) / (max_vals - min_vals)
            radar = normalized_radar
        
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        scaler = MinMaxScaler(feature_range=(0, 0.0005))
        # scaler = StandardScaler()
        # radar = radar.reshape(-1, 1)  # (800, 1)로 변환
        # radar = scaler.fit_transform(radar).reshape(1, -1)  
        radar = torch.Tensor(radar)  
        # radar, scaler.data_min_, scaler.data_max_
        
        if self.sWVD_root_dir is not None:
            # sWVD 데이터 로드
            sWVD = np.load(sWVD_path)
            sWVD = np.abs(sWVD)



            # radar =  torch.concat([radar, sWVD_1D], dim=1)
        
        # 이미지와 레이더 데이터 및 sWVD 데이터 반환 (sWVD 데이터가 없으면 None 반환)
        if self.sWVD_root_dir is not None:
            return image, radar, image_path, radar_path, sWVD
        else:
            return image, radar, image_path, radar_path ,label
        # return image, radar, image_path, radar_path, sWVD if self.sWVD_root_dir is not None else None