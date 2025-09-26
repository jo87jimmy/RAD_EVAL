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
import cv2

import os
import numpy as np  # 用於數值運算


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
        recon_image_tensor, seg_map_logits = model(image_tensor,
                                                   return_feats=False)

    # --- 4. 後處理輸出 ---

    # a. 處理分割圖 (Anomaly Mask)
    seg_map_probs = torch.softmax(seg_map_logits, dim=1)
    # 沿著通道維度 (dim=1) 找到最大機率的索引 (0=正常, 1=異常)
    anomaly_mask_tensor = torch.argmax(seg_map_probs, dim=1)

    # b. 將 Tensor 轉換為可用於顯示的 NumPy Array

    # 原始圖像，調整到與模型輸入相同的尺寸以便比較
    original_image_np = np.array(image.resize(TARGET_SIZE))

    # 反標準化 (De-normalize) 重建圖像，以便能正確顯示
    recon_image_np = recon_image_tensor.squeeze().cpu().numpy().transpose(
        1, 2, 0)
    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)
    recon_image_np = std * recon_image_np + mean
    recon_image_np = np.clip(recon_image_np, 0, 1)  # 將數值限制在 [0, 1] 範圍內

    # 將預測的遮罩轉換為 numpy 格式
    anomaly_mask_np = anomaly_mask_tensor.squeeze().cpu().numpy().astype(
        np.uint8)

    return original_image_np, recon_image_np, anomaly_mask_np


def run_inference(image_path, model, device, save_path, threshold=0.2):
    # ==================================================================
    # 4. 預處理輸入圖像
    # ==================================================================
    print(f"Step 4: Preprocessing the input image: {image_path}..."
          )  # 顯示正在處理的影像路徑

    preprocess = transforms.Compose([
        transforms.Resize([256, 256]),  # 調整影像大小為 256x256
        transforms.ToTensor(),  # 轉換為 Tensor 並標準化到 [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # 依照 ImageNet 均值做正規化
            std=[0.229, 0.224, 0.225])  # 依照 ImageNet 標準差做正規化
    ])

    image = Image.open(image_path).convert("RGB")  # 讀取影像並轉成 RGB
    input_tensor = preprocess(image).unsqueeze(0).to(
        device)  # 增加 batch 維度並搬到 GPU/CPU

    # ==================================================================
    # 5. 前向傳播
    # ==================================================================
    print("Step 5: Performing inference...")  # 顯示正在做推論
    with torch.no_grad():  # 關閉梯度計算，節省記憶體
        recon_image, seg_map = model(input_tensor,
                                     return_feats=False)  # 模型前向傳播，取得重建影像與分割地圖

    # ==================================================================
    # 6. 後處理
    # ==================================================================
    print("Step 6: Post-processing the output...")  # 顯示正在做後處理
    probabilities = torch.softmax(seg_map, dim=1)  # 對分割地圖做 softmax，取得每個類別機率
    anomaly_map = probabilities[:, 1, :, :]  # 取通道1的異常機率 (假設通道0是正常，通道1是異常)

    image_anomaly_score = torch.max(anomaly_map).item()  # 取整張影像最大異常機率作為影像級分數
    print(f"Image-level anomaly score: {image_anomaly_score:.4f}")  # 顯示影像異常分數

    # 二值化 mask
    binary_mask = (anomaly_map > threshold).squeeze().cpu().numpy().astype(
        np.uint8) * 255  # 將異常機率大於閾值的區域設為 255，形成二值化遮罩

    # Normalize anomaly map (0~255)
    anomaly_map_np = anomaly_map.squeeze().cpu().numpy()  # 將異常機率轉成 numpy
    anomaly_map_norm = (anomaly_map_np - anomaly_map_np.min()) / (
        anomaly_map_np.max() - anomaly_map_np.min() + 1e-8)  # 將異常機率正規化到 0~1
    anomaly_map_visual = (anomaly_map_norm * 255).astype(
        np.uint8)  # 將正規化結果轉成 0~255 的影像

    # Heatmap (彩色)
    heatmap = cv2.applyColorMap(anomaly_map_visual,
                                cv2.COLORMAP_JET)  # 將異常地圖轉成彩色熱力圖

    print("Seg_map logits min/max:",
          seg_map.min().item(),
          seg_map.max().item())  # 顯示分割地圖原始 logits 的最小最大值
    print("Anomaly_map min/max:",
          anomaly_map.min().item(),
          anomaly_map.max().item())  # 顯示異常地圖機率的最小最大值

    # ==================================================================
    # 7. 儲存結果
    # ==================================================================
    anomaly_map_path = f"{save_path}_anomaly_map.png"  # 設定異常圖儲存路徑
    binary_mask_path = f"{save_path}_binary_mask.png"  # 設定二值化遮罩儲存路徑
    heatmap_path = f"{save_path}_heatmap.png"  # 設定彩色熱力圖儲存路徑

    Image.fromarray(anomaly_map_visual).save(anomaly_map_path)  # 儲存灰階異常圖
    Image.fromarray(binary_mask).save(binary_mask_path)  # 儲存二值遮罩
    cv2.imwrite(heatmap_path, heatmap)  # 儲存彩色熱力圖

    print(f"✅ 儲存完成：{anomaly_map_path}, {binary_mask_path}, {heatmap_path}"
          )  # 顯示儲存完成訊息

    return anomaly_map_visual, binary_mask  # 回傳異常圖與二值遮罩


def visualize_and_save(original_img, recon_img, anomaly_map, binary_mask,
                       save_path_base):
    """
    將推論結果可視化並儲存成圖片。

    Args:
        original_img (np.ndarray): 原始輸入影像 (H, W, C)。
        recon_img (np.ndarray): 重建後的影像 (H, W, C)。
        anomaly_map (np.ndarray): 異常分數圖 (H, W)，值域 [0, 1]。
        binary_mask (np.ndarray): 二值化的異常遮罩 (H, W)，值為 0 或 255。
        save_path_base (str): 儲存檔案的基礎路徑與檔名 (不含副檔名)。
    """
    # 確保儲存目錄存在
    save_dir = os.path.dirname(save_path_base)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 將 anomaly_map 轉換為熱力圖
    # 將灰階的 anomaly_map (0-1) 轉為 8-bit 整數 (0-255)
    heatmap_gray = (anomaly_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

    # 將熱力圖疊加到原始影像上
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    # 將二值化遮罩轉為三通道，方便合併
    binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # 將四張圖拼接成一張大圖 (原始圖 | 重建圖 | 熱力圖 | 二值圖)
    combined_img = np.hstack(
        [original_img, recon_img, overlay, binary_mask_color])

    # 儲存合併後的影像
    cv2.imwrite(f"{save_path_base}_results.png", combined_img)
    print(f"✅ 結果已儲存至: {save_path_base}_results.png")


def run_inference(img_path, model, device, save_path_base):
    """
    對單張影像執行異常檢測推論。

    Args:
        img_path (str): 輸入影像的路徑。
        model (nn.Module): 訓練好的學生模型。
        device (str): 'cuda' 或 'cpu'。
        save_path_base (str): 儲存結果的基礎路徑與檔名。

    Returns:
        tuple: (anomaly_map, binary_mask)
    """
    # --- 1. 影像預處理 ---
    # 定義與訓練時相同的轉換流程
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 讀取影像並轉換
    original_img_cv = cv2.imread(img_path)
    if original_img_cv is None:
        print(f"❌ 錯誤: 無法讀取影像 {img_path}")
        return None, None
    original_img_rgb = cv2.cvtColor(original_img_cv,
                                    cv2.COLOR_BGR2RGB)  # 轉為 RGB
    img_tensor = transform(original_img_rgb).unsqueeze(0).to(device)

    # --- 2. 執行模型推論 ---
    with torch.no_grad():
        recon_image, seg_map = model(img_tensor)

    # --- 3. 結果後處理 ---
    # 將輸出的 logit 轉換為機率分佈
    seg_map_softmax = torch.softmax(seg_map, dim=1)
    # 取出 "異常" 類別的機率圖 (假設類別 1 是異常)
    anomaly_map_tensor = seg_map_softmax[:, 1, :, :]

    # 將 Tensor 轉為 NumPy array 以便後續處理
    # .squeeze() 去掉批次維度, .cpu() 移至 CPU, .numpy() 轉換格式
    anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()
    recon_image_np = recon_image.squeeze().cpu().permute(1, 2, 0).numpy()
    # 將重建圖的像素值從 [0, 1] 轉回 [0, 255]
    recon_image_np = (recon_image_np * 255).astype(np.uint8)
    recon_image_bgr = cv2.cvtColor(recon_image_np, cv2.COLOR_RGB2BGR)

    # --- 4. 產生二值化遮罩 ---
    # 設定一個閾值，將異常機率大於該值的像素標記為 1 (異常)
    threshold = 0.9
    binary_mask = (anomaly_map > threshold).astype(
        np.uint8) * 255  # 轉為 0 或 255

    # --- 5. 可視化並儲存 ---
    # 確保原始影像尺寸為 256x256，以便拼接
    original_img_resized = cv2.resize(original_img_cv, (256, 256))
    #原始圖、重建圖、熱力圖、二值圖
    visualize_and_save(original_img_resized, recon_image_bgr, anomaly_map,
                       binary_mask, save_path_base)

    return anomaly_map, binary_mask


#完成版!
# def run_inference(image_path, model, device, save_path):
#     # ==================================================================
#     # 4. 預處理輸入圖像
#     #    - 預處理步驟必須與訓練時的驗證集/測試集完全相同！
#     # ==================================================================
#     print(f"Step 4: Preprocessing the input image: {image_path}...")
#     # 這裡的 resize_shape 和 normalize 參數應與訓練時一致
#     preprocess = transforms.Compose([
#         transforms.Resize([256, 256]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224,
#                                   0.225])  # 假設使用 ImageNet 的均值和標準差
#     ])

#     image = Image.open(image_path).convert("RGB")
#     input_tensor = preprocess(image).unsqueeze(0).to(
#         device)  # unsqueeze(0) 是為了增加 batch 維度

#     # ==================================================================
#     # 5. 執行前向傳播 (在 torch.no_grad() 環境下)
#     # ==================================================================
#     print("Step 5: Performing inference...")
#     with torch.no_grad():
#         # 推理時，我們只需要分割圖，所以 return_feats=False
#         # 學生模型會同時輸出重建圖和分割圖
#         recon_image, seg_map = model(input_tensor, return_feats=False)

#     # recon_image = recon_image.squeeze().cpu().numpy().transpose(1, 2, 0)
#     # recon_image = (recon_image * 255).astype(np.uint8)
#     # Image.fromarray(recon_image).save(f"{save_path}_recon.png")
#     # ==================================================================
#     # 6. 後處理輸出結果
#     # ==================================================================
#     print("Step 6: Post-processing the output...")
#     # seg_map 的形狀是 [batch_size, num_classes, H, W]，例如 [1, 2, 256, 256]
#     # 我們需要的是代表 "異常" 的那個通道的機率

#     # 使用 softmax 將 logits 轉換為機率
#     probabilities = torch.softmax(seg_map, dim=1)

#     # 提取異常類別的機率圖 (假設通道 1 代表異常, 通道 0 代表正常)
#     anomaly_map = probabilities[:, 1, :, :]

#     # 獲取整張圖片的異常分數 (可以是最大值或平均值)
#     image_anomaly_score = torch.max(anomaly_map).item()
#     print(f"Image-level anomaly score: {image_anomaly_score:.4f}")

#     # 可以設定一個閾值來得到二值化的異常遮罩
#     threshold = 0.5
#     binary_mask = (anomaly_map
#                    > threshold).squeeze().cpu().numpy().astype(np.uint8)

#     # 將異常分數圖轉換為可視化的灰度圖
#     anomaly_map_visual = (anomaly_map.squeeze().cpu().numpy() * 255).astype(
#         np.uint8)
#     print("Seg_map logits min/max:",
#           seg_map.min().item(),
#           seg_map.max().item())
#     print("Anomaly_map min/max:",
#           anomaly_map.min().item(),
#           anomaly_map.max().item())
#     return anomaly_map_visual, binary_mask


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
        student_model = AnomalyDetectionModel(
            recon_in=IMG_CHANNELS,
            recon_out=IMG_CHANNELS,
            recon_base=64,  # <-- 使用小型模型的參數
            disc_in=IMG_CHANNELS * 2,
            disc_out=SEG_CLASSES,
            disc_base=64  # <-- 使用小型模型的參數
        ).to(device)

        # 載入訓練好的學生模型權重
        model_weights_path = './student_model_checkpoints/bottle.pckl'  # ⬅️ 我的的權重路徑
        student_model.load_state_dict(
            torch.load(model_weights_path, map_location=device))

        # --- 2. 設定為評估模式 ---
        student_model.eval()

        test_path = './mvtec/' + obj_name + '/test'  # 測試資料路徑
        items = ['good', 'broken_large', 'broken_small',
                 'contamination']  # 測試資料標籤
        print(f"🔍 測試資料夾：{test_path}，共 {len(items)} 類別")

        # 依類別逐張讀取影像並執行推論
        for item in items:
            item_path = os.path.join(test_path, item)
            # 建立該類別的輸出資料夾
            output_dir = os.path.join(save_root, obj_name, item)
            os.makedirs(output_dir, exist_ok=True)  # 確保資料夾存在

            if not os.path.exists(item_path):
                print(f"⚠️ 警告: 路徑不存在 {item_path}，跳過。")
                continue

            img_files = [
                f for f in os.listdir(item_path)
                if f.endswith('.png') or f.endswith('.jpg')
            ]

            print(f"\n📂 類別：{item}，共 {len(img_files)} 張影像")

            for img_name in img_files:
                img_path = os.path.join(item_path, img_name)
                print(f"🖼️ 處理影像：{img_path}")

                # 去掉副檔名，只取檔名主體
                base_name, _ = os.path.splitext(img_name)
                # 設定儲存路徑
                save_path_base = os.path.join(output_dir, base_name)

                # --- 執行推理 ---
                anomaly_map, binary_mask = run_inference(
                    img_path, student_model, device, save_path_base)
        print(f"\n✅ 物件類別 {obj_name} 測試完成！")
    print("\n🎉 所有測試已完成！")
    # original, reconstruction, anomaly_mask = predict_anomaly(
    #     student_model, img_path, device)

    # # 將 numpy array 轉換為 PIL Image
    # anomaly_map_img = Image.fromarray(anomaly_map)
    # binary_mask_img = Image.fromarray(binary_mask *
    #                                   255)  # 乘以 255 使其可視化

    # # 儲存圖片，加上原檔名方便區分
    # # 原始輸入路徑
    # orig_img_path = os.path.join(item_path, img_name)
    # save_img_path = os.path.join(save_root, f"{base_name}_img.png")
    # anomaly_map_path = os.path.join(
    #     save_root, f"{base_name}_anomaly_map.png")  #異常圖
    # binary_mask_path = os.path.join(
    #     save_root, f"{base_name}_binary_mask.png")  #異常遮罩
    # Image.open(orig_img_path).save(
    #     save_img_path)  # 開啟原始圖片並另存到 save_root
    # anomaly_map_img.save(anomaly_map_path)
    # binary_mask_img.save(binary_mask_path)

    # print(f"✅ 儲存完成：{anomaly_map_path}, {binary_mask_path}")


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
    parser.add_argument('--gpu_id',
                        action='store',
                        type=int,
                        default=-2,
                        required=False,
                        help='GPU ID (-2: auto-select, -1: CPU)')
    args = parser.parse_args()

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
                 ['transistor'], ['tile'], ['cable'], ['zipper'],
                 ['toothbrush'], ['metal_nut'], ['hazelnut'], ['screw'],
                 ['grid'], ['wood']]

    if int(args.obj_id) == -1:
        obj_list = [
            'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
            'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
            'screw', 'grid', 'wood'
        ]
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
