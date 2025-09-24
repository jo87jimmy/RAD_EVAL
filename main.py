import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import random  # 亂數控制
import argparse  # 命令列參數處理
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, SSIM
from model_unet import AnomalyDetectionModel
from data_loader import MVTecDRAEM_Test_Visual_Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋

# =======================
# Utilities
# =======================
def get_available_gpu():
    """自動選擇記憶體使用率最低的GPU"""
    if not torch.cuda.is_available():
        return -1  # 沒有GPU可用

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1

    # 檢查每個GPU的記憶體使用情況
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_reserved = torch.cuda.memory_reserved(i)
        gpu_memory.append((i, memory_allocated, memory_reserved))

    # 選擇記憶體使用最少的GPU
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def predict_anomaly(model, image_path, device):
    """
    對單張圖片進行異常檢測推論。
    此函數使用固定的、非隨機的預處理流程。

    Args:
        model (nn.Module): 訓練好的學生模型
        image_path (str): 輸入圖片的路徑
        device (str): 'cuda' or 'cpu'

    Returns:
        tuple: (原始圖像, 重建圖像, 異常遮罩) 均為 numpy array
    """
    # --- 1. 定義推論時的圖像預處理流程 ---
    # 這裡只包含必要的、非隨機的轉換。
    # 確保 Resize 的尺寸和 Normalize 的 mean/std 與您訓練時使用的完全一致！

    # 假設您的模型輸入尺寸為 224x224
    TARGET_SIZE = (224, 224)

    # 假設您訓練時使用了 ImageNet 的均值和標準差進行標準化
    # 如果您使用了不同的值，請務必在此處修改！
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    preprocess = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    # --- 2. 載入並預處理圖像 ---
    # 確保以 RGB 模式打開，與訓練時保持一致
    image = Image.open(image_path).convert("RGB")
    # unsqueeze(0) 是為了增加一個批次維度，從 [C, H, W] 變為 [1, C, H, W]
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # --- 3. 執行前向傳播 ---
    with torch.no_grad():
        recon_image_tensor, seg_map_logits = model(image_tensor, return_feats=False)

    # --- 4. 後處理輸出 ---

    # a. 處理分割圖 (Anomaly Mask)
    seg_map_probs = torch.softmax(seg_map_logits, dim=1)
    # 沿著通道維度 (dim=1) 找到最大機率的索引 (0=正常, 1=異常)
    anomaly_mask_tensor = torch.argmax(seg_map_probs, dim=1)

    # b. 將 Tensor 轉換為可用於顯示的 NumPy Array

    # 原始圖像，調整到與模型輸入相同的尺寸以便比較
    original_image_np = np.array(image.resize(TARGET_SIZE))

    # 反標準化 (De-normalize) 重建圖像，以便能正確顯示
    recon_image_np = recon_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)
    recon_image_np = std * recon_image_np + mean
    recon_image_np = np.clip(recon_image_np, 0, 1) # 將數值限制在 [0, 1] 範圍內

    # 將預測的遮罩轉換為 numpy 格式
    anomaly_mask_np = anomaly_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

    return original_image_np, recon_image_np, anomaly_mask_np

# =======================
# Main Pipeline
# =======================
def main(obj_names, args):
    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 建立主存檔資料夾
    save_root = "./save_files"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print("🔄 開始測試，共有物件類別:", len(obj_names))
    for obj_name in obj_names:
        print(f"▶️測試物件類別: {obj_name}")
        # Load
        IMG_CHANNELS = 3
        SEG_CLASSES = 2
        STUDENT_RECON_BASE = 64
        STUDENT_DISC_BASE = 64
        # 實例化學生模型架構
        student_model = AnomalyDetectionModel(
            recon_in=IMG_CHANNELS,
            recon_out=IMG_CHANNELS,
            recon_base=STUDENT_RECON_BASE,
            disc_in=IMG_CHANNELS * 2,
            disc_out=SEG_CLASSES,
            disc_base=STUDENT_DISC_BASE
        ).to(device)

        # 載入訓練好的學生模型權重
        model_weights_path = './student_model_checkpoints/bottle.pckl' # ⬅️ 我的的權重路徑
        student_model.load_state_dict(torch.load(model_weights_path, map_location=device))

        # --- 2. 設定為評估模式 ---
        student_model.eval()

        test_path = './mvtec/' + obj_name + '/test'  # 測試資料路徑
        items = ['good', 'broken_large', 'broken_small', 'contamination']  # 測試資料標籤
        print(f"🔍 測試資料夾：{test_path}，共 {len(items)} 類別")

        # 依類別逐張讀取影像並執行推論
        for item in items:
            item_path = os.path.join(test_path, item)
            img_files = [
                f for f in os.listdir(item_path)
                if f.endswith('.png') or f.endswith('.jpg')
            ]

            print(f"\n📂 類別：{item}，共 {len(img_files)} 張影像")

            for img_name in img_files:
                img_path = os.path.join(item_path, img_name)
                print(f"\n🖼️ 處理影像：{img_path}")
                original, reconstruction, anomaly_mask = predict_anomaly(student_model, img_path, device)

                # --- 可視化結果 ---
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(original)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(reconstruction)
                axes[1].set_title('Reconstructed Image')
                axes[1].axis('off')

                # 將異常遮罩（0和1）與原始圖像疊加顯示
                axes[2].imshow(original)
                axes[2].imshow(anomaly_mask, cmap='jet', alpha=0.4) # 使用半透明疊加
                axes[2].set_title('Anomaly Mask')
                axes[2].axis('off')

                # 儲存整張圖
                plt.tight_layout()
                plt.savefig(f"{save_root}/comparison_{obj_name}_{img_name}.png")
                plt.close()


        # # 建立 dataset / dataloader
        # path = f'./mvtec'  # 測試資料路徑
        # data_dir = os.path.join(path, obj_name, "test")
        # print(f"  📂 建立 dataset: {data_dir}")
        # dataset = MVTecDRAEM_Test_Visual_Dataset(
        #     data_dir, resize_shape=[256,256])
        # dataloader = DataLoader(dataset,
        #                         batch_size=1,
        #                         shuffle=False,
        #                         num_workers=0)
        # print("  ✅ Dataset size:", len(dataset))

        # print("  🚀 開始遍歷 dataloader...")
        # for i_batch, sample_batched in enumerate(dataloader):
        #     # sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}
        #     print(f"    處理 batch {i_batch+1}/{len(dataloader)} (idx={sample_batched['idx'].item()})")
        #     # --- 3. 前處理 ---
        #     gray_batch = sample_batched["image"]

        #     # --- 4. 預測 ---
        #     original, reconstruction, anomaly_mask = predict_anomaly(student_model, gray_batch, device)

        #     # --- 可視化結果 ---
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        #     axes[0].imshow(original)
        #     axes[0].set_title('Original Image')
        #     axes[0].axis('off')

        #     axes[1].imshow(reconstruction)
        #     axes[1].set_title('Reconstructed Image')
        #     axes[1].axis('off')

        #     # 將異常遮罩（0和1）與原始圖像疊加顯示
        #     axes[2].imshow(original)
        #     axes[2].imshow(anomaly_mask, cmap='jet', alpha=0.4) # 使用半透明疊加
        #     axes[2].set_title('Anomaly Mask')
        #     axes[2].axis('off')

        #     # 儲存整張圖
        #     plt.tight_layout()
        #     plt.savefig(f"{save_root}/comparison_{obj_name}_{i_batch}.png")
        #     plt.close()


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    """
    --gpu_id -2：自動選擇最佳GPU
    --gpu_id -1：強制使用CPU
    --gpu_id  0：使用GPU 0（原有行為）
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=-2, required=False,
                    help='GPU ID (-2: auto-select, -1: CPU)')
    args = parser.parse_args()

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [
        ['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
        ['transistor'], ['tile'], ['cable'], ['zipper'], ['toothbrush'],
        ['metal_nut'], ['hazelnut'], ['screw'], ['grid'], ['wood']
    ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule', 'bottle', 'carpet', 'leather', 'pill',
                    'transistor', 'tile', 'cable', 'zipper', 'toothbrush',
                    'metal_nut', 'hazelnut', 'screw', 'grid', 'wood']
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    # 根據選擇的GPU執行
    if args.gpu_id == -1:
        # 使用CPU
        main(picked_classes, args)
    else:
        # 使用GPU
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)
