# Prompted Segmentation for Drywall QA

A text-conditioned segmentation system that generates binary masks for drywall taping areas and wall cracks using natural-language prompts. Built using **CLIP**, **FiLM**, **U-Net**, and **PyTorch**.

---

## ⭐ Project Summary

This project demonstrates a **multi-task, prompt-driven segmentation model** for industrial quality assurance.  
Given an image and a natural-language instruction, the model produces a segmentation mask for:

- **“segment taping area”** → Drywall-Join-Detect dataset  
- **“segment crack”** → Cracks dataset

The model supports flexible inspection workflows and can generalize to new prompts with minimal adjustments.

---

## 📦 Datasets

### **1. Drywall-Join-Detect (Taping area)**  
- 936 train, 250 valid  
- Only YOLO bounding boxes available → converted into binary masks  
- Prompt used: `"segment taping area"`

### **2. Cracks Dataset**  
- 5164 train, 201 valid, 4 test  
- Direct crack segmentation annotations  
- Prompt used: `"segment crack"`

---

## 🧠 Model Architecture

The model combines:

### 🔹 Text Encoder  
- **CLIP ViT-B/32** (frozen)  
- Extracts a 512-dimensional text embedding.

### 🔹 Image Encoder  
- **ResNet-34** pretrained on ImageNet.

### 🔹 Fusion Layer (FiLM)
Feature-wise Linear Modulation applied at each decoder stage:

