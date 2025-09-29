# XAI-Driven Deep Learning Framework for Gastrointestinal Disorder Diagnosis: A-Multi-Stage-Approach

Gastrointestinal (GI) disorders are among the most prevalent health concerns worldwide, and accurate early diagnosis plays a vital role in effective treatment planning. This project introduces a multi-stage deep learning framework enhanced with explainable AI (XAI) to address the challenges of GI disorder classification from endoscopic images.


## Framework Archtecture
<img width="1000" height="300" alt="image" src="https://github.com/user-attachments/assets/d8fd0243-09cc-470a-8078-19fd4dc233d8" />

The framework progressively classifies endoscopic images into:

Stage 1: Coarse localization (Upper GI, Lower GI, Upper–Lower GI)

Stage 2: Intermediate sub-categories (9 anatomical, pathological, and therapeutic groups)

Stage 3: Fine-grained diagnosis (22 disease categories)

Our approach leverages the ConvNeXt-Tiny architecture with Stochastic Weight Averaging (SWA), demonstrating superior accuracy compared to state-of-the-art baselines. To ensure transparency in medical decision-making, we integrate Grad-CAM, saliency maps, and SHAP for model interpretability, highlighting critical visual cues that guide predictions.


## Results

Stage 1: The framework achieved an accuracy of 94.79%, significantly outperforming ResNet-50 (85.80%), DenseNet-121 (90.44%), InceptionResNet-V2 (87.06%), and EfficientNet-B2 (61.06). This demonstrates the strong effectiveness of ConvNeXt-SWA for coarse localization tasks.

Stage 2: At the intermediate level, the framework maintained an accuracy of 85.88%, reflecting robust performance even as classification granularity increased across anatomical, pathological, and therapeutic categories.

Stage 3:Results and detailed model comparisons are illustrated below
<img width="2182" height="661" alt="image" src="https://github.com/user-attachments/assets/d7f0f33d-30d8-47db-bac0-27ffd7445bfc" />


## Explainable AI (XAI) Visualizations
## Grad-CAM , Heatmaps & Saliency Maps
<img width="938" height="236" alt="image" src="https://github.com/user-attachments/assets/8679736d-7fd5-4ac9-870b-165a9b73ee7b" />

## SHAP Explanations
<img width="2632" height="468" alt="image" src="https://github.com/user-attachments/assets/41c3cb12-0386-4899-8154-053423c670e7" />

## Installation
```bash
git clone https://github.com/Pratham061204/XAI-Driven-Deep-Learning-Framework-for-Gastrointestinal-Disorder-Diagnosis-A-Multi-Stage-Approach-
cd XAI-Driven-Deep-Learning-Framework-for-Gastrointestinal-Disorder-Diagnosis-A-Multi-Stage-Approach
pip install -r requirements.txt
```
## Citation
```bash
@article{,
  title={XAI-Driven Deep Learning Framework for Gastrointestinal Disorder Diagnosis: A Multi-Stage Approach},
  author={Pratham Chanchlani, Dr. Sapna Sadhwani},
  journal={To Appear},
  year={2025}
}
```

