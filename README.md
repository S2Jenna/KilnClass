## Kiln Classification – Model Pipeline
This project solves a binary satellite image classification task to identify zigzag kilns using transfer learning with ResNet50 in PyTorch.

### Model
- Architecture: ```ResNet50```
- Pretrained on **ImageNet**
- Final fully connected layer replaced with a single-output linear layer to predict the logit for the “zigzag kiln” class

### Training Pipeline
- Image preprocessing and augmentation applied via PyTorch ```DataLoader```
- Loss function: ```BCEWithLogitsLoss```
- Optimizer: ```AdamW```
- Learning-rate scheduler
- Early stopping (patience = 5)
- Model selection based on validation AUC
- Final model retrained on 100% of training data for a fixed number of epochs

### Evaluation
- Performance monitored using validation AUC
- Final predictions generated from the trained ResNet50 model
- Prediction file submitted to the leaderboard for public and private evaluation

### Results
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/ce0fae6f-e235-4e04-8563-e2dfc24564c1" />

- Public leaderboard AUC: 99.94%
- Private leaderboard AUC: 99.86%
ResNet50 achieved strong generalization due to its deep residual architecture, making it well suited for capturing spatial patterns in satellite imagery.
