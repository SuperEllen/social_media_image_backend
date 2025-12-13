"""
Image Feature Extractor

This module extracts visual features from images for engagement prediction.
Currently implements basic statistical features - can be extended with
Vision Transformer or other ML models.
"""

import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict, Any
from colorsys import rgb_to_hsv
from transformers import CLIPImageProcessor, CLIPModel
from scipy.stats import entropy
import torch

# Warm_Hue_Proportion + Average_Saturation + Average_Brightness + Contrast_of_Brightness + Proportion_Brightness
def calculate_image_metrics(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = np.array(image) / 255.0  # 归一化RGB值到[0, 1]

    # 初始化指标
    warm_hue_proportion = 0
    total_saturation = 0
    total_brightness = 0
    bright_pixel_count = 0
    pixel_count = image.shape[0] * image.shape[1]
    
    brightness_list = []

    for pixel in image.reshape(-1, 3):  # 每个像素逐一处理
        r, g, b = pixel
        h, s, v = rgb_to_hsv(r, g, b)
        
        # 1. 计算暖色比例：黄色和红色通常对应0到60度（0到1的h值）
        if 0 <= h <= 1/6 or 5/6 <= h <= 1:
            warm_hue_proportion += 1

        # 2. 累加饱和度和亮度
        total_saturation += s
        total_brightness += v
        brightness_list.append(v)

        # 3. 统计亮度大于0.7的像素
        if v > 0.7:
            bright_pixel_count += 1

    # 计算各个指标
    warm_hue_proportion /= pixel_count
    avg_saturation = total_saturation / pixel_count
    avg_brightness = total_brightness / pixel_count
    contrast_of_brightness = np.std(brightness_list)
    bright_pixel_proportion = bright_pixel_count / pixel_count

    return warm_hue_proportion, avg_saturation, avg_brightness, contrast_of_brightness, bright_pixel_proportion

def preprocess_image(image_bytes, processor):
    """对图片进行预处理"""
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    print('testing')
    return processor(images=image, return_tensors="pt", padding=True)

def get_attention_entropy_all_layer(outputs):
    # 拿到总的层数
    num_layers = len(outputs.attentions)
    cls_attentions = []
    attention_absolutes = []
      
    # 每次加一的循环应该用range()
    for layer_idx in range(num_layers):
        attention = outputs.attentions[layer_idx][0]  # shape: (num_heads, num_patches+1, num_patches+1)
        cls_attention = attention[:, 0, :]  # CLS 向量的注意力分布，shape: (num_heads, num_patches+1)
        cls_attentions.append(cls_attention.mean(dim=0).detach().cpu().numpy())
    # 计算 attention_entropy
    attention_entropies = [entropy(cls_attention) for cls_attention in cls_attentions]
    return attention_entropies


def get_image_AE(image_bytes: bytes):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", output_attentions = True, attn_implementation="eager")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    inputs = preprocess_image(image_bytes, processor)

    # outputs = model.vision_model(pixel_values=inputs["pixel_values"],output_attentions=True,return_dict=True,output_hidden_states=True)
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"],output_attentions=True,return_dict=True,output_hidden_states=True)
    attention_entropies = get_attention_entropy_all_layer(outputs)
    low_level_attention_entropy = np.mean(attention_entropies[1:6])
    mid_level_attention_entropy = np.mean(attention_entropies[7:9])
    high_level_attention_entropy = np.mean(attention_entropies[9:])
    return low_level_attention_entropy, mid_level_attention_entropy, high_level_attention_entropy


def extract_image_features(image_bytes: bytes) -> Dict[str, Any]:
    warm_hue_proportion, avg_saturation, avg_brightness, contrast_of_brightness, bright_pixel_proportion = calculate_image_metrics(image_bytes)
    low_level_attention_entropy, mid_level_attention_entropy, high_level_attention_entropy = get_image_AE(image_bytes)
    
    return {
        "Warm_Hue_Proportion": warm_hue_proportion,
        "Average_Saturation": avg_saturation,
        "Average_Brightness": avg_brightness,
        "Contrast_of_Brightness": contrast_of_brightness,
        "Proportion_Brightness": bright_pixel_proportion,
        "Low_Level_Attention_Entropy": low_level_attention_entropy,
        "Mid_Level_Attention_Entropy": mid_level_attention_entropy,
        "High_Level_Attention_Entropy": high_level_attention_entropy
    }
