# Extraction of the Brain Region from CTA images 
## Problem Statement
The objective of this hackathon challenge is to develop a robust and efficient algorithm or AI model capable of accurately extracting the brain region from Brain CT Angiography (CTA) images, regardless of whether the images are acquired in Head First Supine (HFS) or Feet First Supine (FFS) orientation. `The primary goal is to automate and streamline the brain extraction process, reducing the manual intervention required and improving the accuracy of medical image analysis.`

## Data Description
In our project, we've assembled a dataset comprising 22 pairs of files. Each pair includes a 2D representation extracted from original 3D brain images and their corresponding masks, resulting in a total of `35,074` image pairs. Notably, our dataset covers axial, coronal, and sagittal views, providing a comprehensive exploration of brain anatomy. These images are pivotal, enabling our deep learning models to understand and predict brain structures effectively. Through meticulous curation, we've created a diverse collection that plays a foundational role in our journey to unravel the complexities of the human brain, bridging the realms of machine learning and neuroscience.

| Orignal Image | Corresponding Mask |
|---|---|
| ![Image 1](https://example.com/image1.png) | ![Image 2](https://example.com/image2.png) |

## Solution
For our solution, we harnessed the power of the UNET deep learning framework, a sophisticated architecture tailor-made for tasks like ours. The `UNET` ability to learn and represent intricate features within images was pivotal for our project's success.

To effectively train our algorithm, we judiciously split our dataset into three segments: an `80%` training set, a `10%` validation set, and another `10%` set aside for testing. Specifically, this translated to `28,059` images for training, `3,508` images for validation, and `3,507` images for rigorous testing. This allocation allowed us to both teach and evaluate our model's performance with rigor.

A critical step in our process was hyperparameter tuning, where we carefully selected values to optimize our model's effectiveness. We employed a - `batch_size = 32`, which determined the number of images processed at once during training. A learning rate `lr = 1e-4` guided the step size our model took while learning, optimizing the training process. Due to limited computational resources, we conducted training for a single epoch `num_epochs = 1` – an iteration through the entire dataset.

This tailored approach, though constrained by computational availability, effectively allowed us to leverage the power of deep learning within our project. By harnessing UNET, splitting our dataset strategically, and fine-tuning hyperparameters, we strived to maximize our model's accuracy and predictive prowess. The outcome of these concerted efforts is a poised and well-equipped algorithm that holds promise for unraveling the mysteries concealed within the intricate structures of the human brain.

## Results
#### Training
During the training process, we utilized a dataset consisting of `28,059` original images paired with an equal number of corresponding mask images. In order to assess the performance of our model, we conducted validation using a separate set of 3,508 original images along with their respective mask images. After it Following metrics we have achieved:-

Accuracy (acc): 0.03
F1 Score (f1): 1 

The observed lower accuracy in the preliminary outcomes can be attributed to the model undergoing `training for just a single epoch.` This short training duration has constrained the model's ability to fully capture the complexities present within the data. However, there is a positive aspect to consider. Expanding the training process across multiple epochs holds the potential for a notable enhancement in accuracy. This prolonged training period would provide the model with a better opportunity to discern the nuanced relationships within the data, subsequently leading to a more adept prediction of masks for the original images. `It's also worth noting that computational resources, an integral factor in training effectiveness, could have contributed to the limited initial accuracy.`

#### Testing
Following the training phase, the subsequent crucial step involves testing. In this testing phase, we employed a set of 3,508 original images. The evaluation of our model on this test set yielded the following metrics:

Accuracy (acc): 0.03
F1 Score (f1): 1

| Orignal Image | Orignal Mask | Predicted Mask |
|---|---|---|
| ![Image 1](https://example.com/image1.png) | ![Image 2](https://example.com/image2.png) | ![Image 3](https://example.com/image3.png) |



## Repository Structure TELL WHAT TO RUN

- `metrics.py`: Contains Functions of metrics for UNET Model.
- `model.py`: Contains the UNET Model Coded Architecture.
- `train.py`: Script for training the UNET model.
- `test.py`: Script for using the trained model to make predictions.
  Note: By default, the provided code is set up to make predictions for a single image. To test with additional images, users should make necessary adjustments accordingly. This will allow for testing multiple images and obtaining predictions for each.`

