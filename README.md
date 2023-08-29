# Extraction of the Brain Region from CTA images 
## Problem Statement
The objective of this hackathon challenge is to develop a robust and efficient algorithm or AI model capable of accurately extracting the brain region from Brain CT Angiography (CTA) images, regardless of whether the images are acquired in Head First Supine (HFS) or Feet First Supine (FFS) orientation. `The primary goal is to automate and streamline the brain extraction process, reducing the manual intervention required and improving the accuracy of medical image analysis.`

## Data Description
In our project, we've assembled a dataset comprising 22 pairs of files. Each pair includes a 2D representation extracted from original 3D brain images and their corresponding masks, resulting in a total of `35,074` image pairs. Notably, our dataset covers axial, coronal, and sagittal views, providing a comprehensive exploration of brain anatomy. These images are pivotal, enabling our deep learning models to understand and predict brain structures effectively. Through meticulous curation, we've created a diverse collection that plays a foundational role in our journey to unravel the complexities of the human brain, bridging the realms of machine learning and neuroscience.

## Solution
For our solution, we harnessed the power of the UNET deep learning framework, a sophisticated architecture tailor-made for tasks like ours. The `UNET` ability to learn and represent intricate features within images was pivotal for our project's success.

To effectively train our algorithm, we judiciously split our dataset into three segments: an `80%` training set, a `10%` validation set, and another `10%` set aside for testing. Specifically, this translated to `28,059` images for training, `3,508` images for validation, and `3,507` images for rigorous testing. This allocation allowed us to both teach and evaluate our model's performance with rigor.

A critical step in our process was hyperparameter tuning, where we carefully selected values to optimize our model's effectiveness. We employed a - `batch_size = 32`, which determined the number of images processed at once during training. A learning rate `lr = 1e-4` guided the step size our model took while learning, optimizing the training process. Due to limited computational resources, we conducted training for a single epoch `num_epochs = 1` – an iteration through the entire dataset.

This tailored approach, though constrained by computational availability, effectively allowed us to leverage the power of deep learning within our project. By harnessing UNET, splitting our dataset strategically, and fine-tuning hyperparameters, we strived to maximize our model's accuracy and predictive prowess. The outcome of these concerted efforts is a poised and well-equipped algorithm that holds promise for unraveling the mysteries concealed within the intricate structures of the human brain.
