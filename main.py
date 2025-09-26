import os
import torch
from torchvision import transforms as T
import numpy as np
import random  # 亂數控制
import argparse  # 命令列參數處理
from model_unet import AnomalyDetectionModel
import torchvision.transforms as transforms
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
    threshold = 0.5
    binary_mask = (anomaly_map > threshold).astype(
        np.uint8) * 255  # 轉為 0 或 255

    # --- 5. 可視化並儲存 ---
    # 確保原始影像尺寸為 256x256，以便拼接
    original_img_resized = cv2.resize(original_img_cv, (256, 256))
    #原始圖、重建圖、熱力圖、二值圖
    visualize_and_save(original_img_resized, recon_image_bgr, anomaly_map,
                       binary_mask, save_path_base)

    return anomaly_map, binary_mask


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
