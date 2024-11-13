import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import os

# 设置图像的路径和训练参数
image_dir = './training_images' 
output_dir = './generated_faces'
batch_size = 8
epochs = 100
learning_rate = 1e-4

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# 加载图像数据
class FacialDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform):
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

dataset = FacialDataset(image_dir, transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义高对比度噪声生成器
def high_contrast_noise(image_size):
    noise = torch.randn(image_size)
    bright_noise = noise * 1.5  # 提升亮区噪声
    dark_noise = noise * 0.5    # 减弱暗区噪声
    mask = torch.bernoulli(torch.full(image_size, 0.5))  # 随机生成亮暗区域
    return mask * bright_noise + (1 - mask) * dark_noise

# 定义带有Attention机制的卷积块
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # 定义一系列卷积和激活层
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# 定义带有Attention机制的U-Net模型
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AttentionUNet, self).__init__()
        
        # 下采样路径 (编码器)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 最底层卷积
        self.bottleneck = self.conv_block(512, 1024)

        # 上采样路径 (解码器)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder1 = self.conv_block(128, 64)

        # 最终输出卷积层
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    # 卷积块定义
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # 前向传播
    def forward(self, x):
        # 编码路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # 最底层
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # 解码路径并加入Attention模块
        dec4 = self.upconv4(bottleneck)
        att4 = self.att4(dec4, enc4)
        dec4 = torch.cat((att4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        att3 = self.att3(dec3, enc3)
        dec3 = torch.cat((att3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        att2 = self.att2(dec2, enc2)
        dec2 = torch.cat((att2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        att1 = self.att1(dec1, enc1)
        dec1 = torch.cat((att1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        # 最终输出
        return self.output_conv(dec1)

u_net = AttentionUNet()

# 定义Stable Diffusion模型
class StableDiffusionModel(nn.Module):
    def __init__(self, unet):
        super(StableDiffusionModel, self).__init__()
        self.unet = unet

    def forward_diffusion(self, x, timesteps):
        return x * torch.sqrt(timesteps) + high_contrast_noise(x.shape) * torch.sqrt(1 - timesteps)

    def reverse_diffusion(self, x, timesteps):
        return x / torch.sqrt(timesteps) - self.unet(x) * torch.sqrt(1 - timesteps)

model = StableDiffusionModel(u_net)

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 模型训练
for epoch in range(epochs):
    for images in data_loader:
        noisy_images = model.forward_diffusion(images, torch.rand(images.shape))
        
        outputs = model.reverse_diffusion(noisy_images, torch.rand(images.shape))
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 测试代码生成脸谱
def generate_facial_makeup(prompt="京剧脸谱"):
    model.eval()
    with torch.no_grad():
        # 使用CLIP模型将文本提示转换为特征（此部分可根据需要修改）
        text_features = torch.randn((1, 512))  # 使用随机文本特征替代
        x = torch.randn((1, 3, 512, 512))  # 从噪声图像开始

        # 逐步进行反向扩散
        for i in range(50):  # 可调整扩散步数
            x = model.reverse_diffusion(x, torch.tensor([i/50.0]))

        # 将生成图像保存
        generated_image = transforms.ToPILImage()(x.squeeze())
        output_path = os.path.join(output_dir, f"{prompt}_generated.png")
        generated_image.save(output_path)
        print(f"生成的脸谱图像已保存到: {output_path}")

# 调用测试生成函数
generate_facial_makeup()
