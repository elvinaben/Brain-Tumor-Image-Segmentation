# Brain Tumor Image Segmentation

## Project Overview
The "Brain Tumor Image Segmentation" project aims to develop a deep learning model for segmenting brain MRI images, focusing on accurately identifying tumor regions at the pixel level. This automated segmentation enhances the ability of medical professionals to diagnose and plan treatments for brain tumors, contributing to better patient outcomes.

## Background
Early detection of brain tumors is critical for effective treatment. MRI imaging is essential for this, but distinguishing tumor tissue from healthy tissue is often challenging. By leveraging deep learning techniques, this project aims to automate tumor detection, offering healthcare professionals reliable assistance in their diagnostic efforts, thereby improving the quality of patient care.

## Dataset
The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), contains 2,146 brain MRI images, split into 70% for training, 20% for validation, and 10% for testing. Each image is 640x640 pixels and categorized into two classes:
- **Class 0 (Non-Tumor):** Pixels without tumor tissue.
- **Class 1 (Tumor):** Pixels indicating tumor presence.

<img width="885" alt="image" src="https://github.com/user-attachments/assets/32957aaf-acde-4f5d-8cdd-914960914459">

The six sample images below show MRI scans with red-bordered squares indicating tumor areas. These areas are the target for the segmentation model.

## Data Preprocessing
1. Convert JSON files to segmentation masks.
2. Apply histogram equalization to enhance image contrast.
3. Resize all images and masks to 224x224 pixels.
4. Transform images and masks into tensors for deep learning model input.

## Model Configuration
All models share the following configurations:
- **Loss Function:** Dice Loss + Binary Crossentropy
- **Input:** Channels = 1, Classes = 1
- **Optimizer:** Adam
- **Device:** CUDA
- **Epochs:** 35

### Model 1: U-Net with Unfrozen Encoder Layers  
- **Encoder:** EfficientNet-B0 (pre-trained on ImageNet, unfrozen layers)
- **Decoder:** U-Net
- **Additional Layers:** 
  - Batch Normalization in the encoder.
  - Dropout (0.5) in the decoder and classification head.
- **Learning Rate:** 0.001

### Model 2: U-Net with Frozen Encoder Layers  
- **Encoder:** EfficientNet-B0 (pre-trained on ImageNet, frozen layers)
- **Decoder:** U-Net
- **Additional Layers:** 
  - Batch Normalization in the encoder.
  - Dropout (0.5) in the decoder and classification head.
- **Learning Rate:** 0.001

### Model 3: Efficient U-Net++ with Unfrozen Encoder Layers  
- **Encoder:** EfficientNet-B5 (pre-trained on ImageNet, unfrozen layers)
- **Decoder:** U-Net++
- **Learning Rate:** 0.0001

## Evaluation
The models were evaluated using Dice Loss and Intersection over Union (IoU), both metrics indicating segmentation performance.

![Evaluation Table](https://github.com/user-attachments/assets/5cf05fd5-9d6b-4798-a2e2-7c1723c9ff0d)  

### Explanation:
Model 3 (Efficient U-Net++) outperformed the others with the lowest Dice Loss of 0.3057 and the highest IoU of 0.5426. The Dice Loss measures the overlap between the predicted and ground truth masks, with lower values indicating better performance. An IoU of 0.5426 means that approximately 54.26% of the predicted mask overlaps with the ground truth, highlighting Model 3 as the best performing model.

### Model 3 Inference
Below is the result of Model 3's prediction for brain tumor presence based on MRI images. The prediction can be compared with the ground truth, which is rectangular in shape.

<img width="668" alt="image" src="https://github.com/user-attachments/assets/74ef7eb5-7f48-4f28-8bcd-0618362b43be">


## Conclusion
Model 3, Efficient U-Net++, demonstrated the best performance, with a Dice Loss of 0.3057 and an IoU of 0.5426. However, all models suffered from overfitting and struggled to accurately segment MRI images that were almost completely white. This issue likely stems from the dataset, which contains MRI images with minimal pixel variation. Additionally, the rectangular ground truth masks presented challenges for semantic segmentation, contributing to higher loss values. 

Future improvements could involve exploring object detection methods, refining model architectures, and utilizing more advanced data augmentation techniques to further enhance model performance and address these challenges.

