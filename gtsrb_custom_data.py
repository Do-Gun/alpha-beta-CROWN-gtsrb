# 📝 저장 위치:
# alpha-beta-CROWN/complete_verifier/custom/gtsrb_custom_data.py

import torch
import torch.nn as nn
import os
from torchvision.datasets import GTSRB
from torchvision import transforms
from torch.utils.data import DataLoader

# ----------------- 모델 구조 정의 -----------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------- alpha-beta-CROWN이 호출할 함수들 -----------------

def gtsrb_model(**kwargs):
    """
    SimpleCNN 모델 구조를 생성하여 반환하는 함수.
    YAML의 'name' 필드에서 이 함수를 호출합니다.
    """
    model = SimpleCNN()
    return model

def gtsrb_dataloader(spec, **kwargs):
    """
    GTSRB 테스트 데이터와 강인성(robustness) 경계를 튜플 형태로 반환하는 함수.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # complete_verifier 폴더를 기준으로 상대 경로를 설정합니다.
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
    test_dataset = GTSRB(root=database_path, split='test', transform=transform, download=True)
    
    # 마감 시간이 임박했으므로, 10개의 샘플만 테스트합니다.
    num_samples = 1
    
    # 데이터셋에서 10개의 이미지를 가져와 하나의 텐서로 합칩니다.
    images = torch.stack([test_dataset[i][0] for i in range(num_samples)])
    labels = torch.tensor([test_dataset[i][1] for i in range(num_samples)])

    # 강인성(robustness) 경계를 계산합니다.
    # ToTensor()만 사용했으므로, 데이터는 0~1 범위에 있습니다.
    eps = spec["epsilon"]
    
    # 엡실론(epsilon)이 적용된 데이터의 상한과 하한을 계산합니다.
    # clamp 함수를 사용하여 값이 0과 1 사이를 벗어나지 않도록 합니다.
    data_max = torch.clamp(images + eps, 0, 1)
    data_min = torch.clamp(images - eps, 0, 1)

    # 🚨핵심 수정 사항: AssertionError를 해결하기 위해 5번째 반환값을 None이 아닌 eps 값으로 수정합니다.
    return images, labels, data_max, data_min, eps
