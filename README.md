# Multi-Head CNN Model for CIFAR-100 Classification

## Introduction
This project involves implementing a multi-head convolutional neural network (CNN) from scratch using Python to classify images from the CIFAR-100 dataset. The model is designed to handle multiple classification tasks simultaneously by leveraging separate output heads for different class subsets.

## Objective
The key objectives of this project include:
- Designing and implementing a CNN with multiple classification heads.
- Training and evaluating the model on the CIFAR-100 dataset.
- Comparing different optimization strategies and activation functions.
- Analyzing performance using accuracy, loss curves, and confusion matrices.
- Reporting the total trainable parameters.

## Dataset
The CIFAR-100 dataset consists of 60,000 color images (32×32 pixels) across 100 classes, grouped into 20 superclasses. The dataset is split into:
- **Training set:** 50,000 images
- **Test set:** 10,000 images

### Preprocessing Steps:
1. **Normalization:**
   - Pixel values are scaled to [0,1] for better convergence.
2. **Data Augmentation:**
   - Techniques like random cropping and flipping applied.
3. **One-hot Encoding:**
   - Labels are converted into one-hot vectors.

## Neural Network Architecture
The multi-head CNN consists of the following layers:
1. **Convolutional Layers:**
   - Multiple conv layers extract hierarchical features.
   - ReLU activation used in intermediate layers.
2. **Pooling Layers:**
   - Max pooling reduces spatial dimensions.
3. **Fully Connected Layers:**
   - Extract high-level representations.
4. **Output Heads:**
   - Separate classification heads for different class subsets (fine, superclass, and group-level labels).

### Initialization Strategy:
- Weights initialized using **He Initialization**.
- Bias values initialized to **0**.

## Implementation Details
### Forward Propagation:
- Feature extraction via CNN layers.
- Separate forward passes for each classification head.

### Loss Function:
- **Categorical Cross-Entropy Loss** used for multi-class classification.

### Backward Propagation:
- Gradients computed using backpropagation.
- Optimization performed using **Adam and SGD**.

### Regularization:
- **Dropout and Batch Normalization** applied to prevent overfitting.

## Evaluation Metrics
1. **Accuracy and Loss Curves:**
   - Tracked training and validation accuracy/loss for 50 epochs.
2. **Confusion Matrices:**
   - Generated for test sets to analyze misclassifications.

## Experimental Configurations
1. **Optimization Strategies Compared:**
   - **Adam** vs **SGD**.
2. **Activation Functions:**
   - **ReLU** vs **Leaky ReLU**.
3. **Train-Test Splits:**
   - Experiments performed with **80:20 and 90:10** splits.
4. **Total Trainable Parameters:**
   - The network consists of **2.5 million** trainable parameters.

## Results
| Optimizer | Activation | Final Test Accuracy |
|-----------|-----------|--------------------|
| **Adam**  | **ReLU**  | **72.5%**         |
| **SGD**   | **Leaky ReLU** | **70.8%**         |

![acc](https://github.com/user-attachments/assets/c8bc6aa9-2294-4678-b86b-4f8721538746)
![loss](https://github.com/user-attachments/assets/28d05550-5c24-42cf-967e-d1b6d6a10bd4)
![cf group](https://github.com/user-attachments/assets/e5a60066-c057-4349-beb3-73f980a5e555)
![cf coar](https://github.com/user-attachments/assets/b21ee16d-c83c-43fc-a633-9c8ede7ecb11)
![cf fine](https://github.com/user-attachments/assets/69fed096-c349-44b7-a51b-7485e18b8414)

### Observations
- **Adam optimizer** led to faster convergence.
- **ReLU performed slightly better** than Leaky ReLU.
- **Confusion matrices** showed class imbalance issues.
- **Batch Normalization** helped stabilize training.

## Bonus Task: Severity-Based Misclassification Penalty
### Objective
A custom loss function was implemented to penalize misclassifications based on severity levels:
- **Severity 1:** Misclassified within the same superclass.
- **Severity 2:** Misclassified within the same group but different superclass.
- **Severity 3:** Misclassified in a different group.

### Custom Loss Function
The loss function was modified as follows:
```python
Loss = CrossEntropyLoss(y, y_pred) * (alpha + severity)
```
where:
- `CrossEntropyLoss(y, y_pred)`: Base classification loss.
- `alpha`: A base scaling factor.
- `severity`: The penalty assigned based on misclassification severity level.

### Results and Analysis
After applying the severity-based loss function, the model showed:
- Improved classification accuracy (**70.36%** on fine classes).
- A reduction in high-severity misclassifications.

| Observation | Before | After |
|-------------|--------|-------|
| Correct classifications | 6722 | 7036 |
| Severity 1 errors | 154 | 144 |
| Severity 2 errors | 259 | 201 |
| Severity 3 errors | 2865 | 2619 |

![bonus ac loss](https://github.com/user-attachments/assets/db863e4f-b172-42ac-8e2f-f0a3926949a9)
![bonus cf](https://github.com/user-attachments/assets/4412ba79-b1ac-40e0-80c8-6fc469b7fd98)


These results indicate that incorporating misclassification severity into the training process reduces critical errors and improves overall classification reliability.

## Conclusion
This project successfully implemented a **multi-head CNN model** for CIFAR-100 classification, achieving over **72% accuracy**. The study compared **Adam vs SGD**, experimented with **activation functions**, and analyzed **confusion matrices**. The **bonus task** demonstrated that severity-based loss modification improves classification reliability.

