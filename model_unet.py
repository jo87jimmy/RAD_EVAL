import torch
import torch.nn as nn


# ==============================================================================
# 1. 建立一個統一的模型結構，將重建和判別網路組合起來
#    這個結構將同時用於教師和學生模型
# ==============================================================================
class AnomalyDetectionModel(nn.Module):

    def __init__(self, recon_in, recon_out, recon_base, disc_in, disc_out,
                 disc_base):
        super(AnomalyDetectionModel, self).__init__()
        # 初始化異常檢測模型，包含重建子網路與判別子網路

        # --- 重建子網路 (對應圖中的 TRecon / SRecon) ---
        self.reconstruction_subnet = ReconstructiveSubNetwork(
            in_channels=recon_in,  # 重建子網路輸入通道數
            out_channels=recon_out,  # 重建子網路輸出通道數
            base_width=recon_base  # 重建子網路基礎通道寬度
        )

        # --- 判別子網路 (對應圖中的 TDisc / SDisc) ---
        self.discriminator_subnet = DiscriminativeSubNetwork(
            in_channels=disc_in,  # 判別子網路輸入通道數
            out_channels=disc_out,  # 判別子網路輸出通道數 (通常是 anomaly map)
            base_channels=disc_base  # 判別子網路基礎通道寬度
        )

    def forward(self, x, return_feats=False):
        # 定義前向傳播 (forward pass)

        # --- 重建分支 ---
        # Input --> TRecon/SRecon --> TReconOut/SReconOut
        recon_image = self.reconstruction_subnet(x)  # 經過重建子網路，得到重建影像

        # --- 判別分支 ---
        # TReconOut/SReconOut --> TCat/SCat <-- Input
        # 注意：判別網路的輸入是原圖與重建圖的拼接 (在通道維度上拼接)
        disc_input = torch.cat((x, recon_image), dim=1)

        # TCat/SCat --> TDisc/SDisc --> (TSegMap/SSegMap, TFeatures/SFeatures)
        if return_feats:
            # 如果需要返回特徵 (features)，則輸出分割圖與中間特徵
            seg_map, features = self.discriminator_subnet(disc_input,
                                                          return_feats=True)
            return recon_image, seg_map, features
        else:
            # 僅輸出分割圖 (異常區域對應的 segmentation map)
            seg_map = self.discriminator_subnet(disc_input, return_feats=False)
            return recon_image, seg_map


class ReconstructiveSubNetwork(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        # EncoderReconstructive 的 forward 必須返回一個特徵列表
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width,
                                             out_channels=out_channels)

    def forward(self, x):
        # 注意：不再需要 return_feats，因為蒸餾不在這裡進行
        features = self.encoder(x)
        # 解碼器只使用最後一層特徵來重建圖像
        output = self.decoder(features[-1])
        return output


class DiscriminativeSubNetwork(nn.Module):

    def __init__(self, in_channels=6, out_channels=2, base_channels=64):
        super(DiscriminativeSubNetwork, self).__init__()
        # 注意：in_channels 現在是 6 (例如 3通道原圖 + 3通道重建圖)
        # 注意：out_channels 現在是 2 (異常/正常的 logits)
        self.encoder_segment = EncoderDiscriminative(in_channels,
                                                     base_channels)
        self.decoder_segment = DecoderDiscriminative(base_channels,
                                                     out_channels=out_channels)

    def forward(self, x, return_feats=False):
        # 根據圖表，這裡提取多層次特徵圖
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        output_segment = self.decoder_segment(b1, b2, b3, b4, b5, b6)

        if return_feats:
            # 返回分割圖和用於蒸餾的中間特徵
            features = [b1, b2, b3, b4, b5, b6]
            return output_segment, features
        else:
            return output_segment


class EncoderDiscriminative(nn.Module):

    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        # ... (內部程式碼與您提供的相同) ...
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):

    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8),
                      base_width * 8,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8),
                      base_width * 4,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4),
                      base_width * 2,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1),
                      base_width,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out


class EncoderReconstructive(nn.Module):

    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        # ✅ 正確存通道數
        self.out_channels = [
            base_width, base_width * 2, base_width * 4, base_width * 8,
            base_width * 8
        ]

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        # return b5
        return [b1, b2, b3, b4, b5]  # ⬅️ 回傳多層特徵


class DecoderReconstructive(nn.Module):

    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 1, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 1),
            nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
        #self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):
        up1 = self.up1(b5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out
