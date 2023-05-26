Real and Fake Face Detection (RFFD)
---------------------------

<p style="text-align: justify">
In this project, we use Deep Neural Networks to identify which image is fake or real. The training will be done on a dataset that we got from Kaggle (check it here <a href="https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection?resource=download)">kaggle_real_fake_faces</a>) created by <i style="color:chocolate">Seonghyeon Nam, Seoung Wug Oh, et al.</i> They used expert knowledge to photoshop authentic images. The fake images range between easy, medium, or hard to recognize. The modifications are made on the eyes, nose, and mouth (which permit human beings to recognize others) or the whole face.

</p>

![fake_photoshop](https://github.com/minostauros/Real-and-Fake-Face-Detection/raw/master/filename_description.jpg)

<i style="color: lightgray">The above image was got from the kaggle description of the image and describe  a file.</i>

The image above is described as a fake image file. The name of the file can be decomposed into three different parts separated by underscores:

- The first part indicates the quality of the Photoshop or the difficulty of recognizing that it is fake;
- The second part indicates the identification number of the image;
- The third and final part indicates the modified segment of the face in binary digits with the following signature -> 
```json
[bit_left_eye, bit_right_eye, bit_nose, bit_mouth]
```

The segment is modified if it is the positive bit (1). Otherwise, the segment is not modified. 

Let us define bellow, with more details, our main objective.

### Objective 

The purpose of the project is to use `Vision Transformer (ViT)` mixed with `Transfer Learning` to achieve a great accuracy and recall on the validation set. ViT is a new field which try to reproduce the same performance that the Convolution Neural Networks on image classification task but using the Transformer architecture. It can provide very accurate results. 

![VISION_TRANSFORMER](https://www.researchgate.net/profile/Jacob-Heilmann-Clausen/publication/357885173/figure/fig1/AS:1113907477389319@1642587646516/Vision-Transformer-architecture-main-blocks-First-image-is-split-into-fixed-size_W640.jpg)

However, we cannot obtain such great result with only few images. ViT require around 14 millions images to learn on image classification task and we want to train the model only on one GPU device. Then the solution is to use Transfer Learning with a pre-trained Transformer to improve the overall performance. 

We will fine-tune the pre-trained ViT Model for which the ArXiv paper is available at the following link [Vision Transformer](https://arxiv.org/pdf/2010.11929). It was pre-trained on the ImageNet-21k which contains 14 millions of images distributed over 21 thousand classes. The model is available in HuggingFace and can be import with the HuggingFace API.

For the moment, we want to obtain the following scores on the validation set:

- **Accuracy > 80**
- **f1 > 80**

Since it is related to medical predictions but only on social network security managing we will enforce ourself to obtain not search for more than 90% of **Accuracy** and **f1-score**.

The next section describe the steps that are required to achieve the project.

### Steps 🧾

Let us define bellow the main parts of our project:

- Data generation and exploration: We must recuperate the images and visualize them and identify their statistics. Moreover we will identify the augmentation methods to add to the images ➡️ [Generating_and_visualizing](../generate_and_visualize.md)

- Preprocessing method: We must after exploration define the preprocessing to add before training the model on them. ➡️ [Preprocessing_and_loading](../preprocessing_and_loading.md)

- Split the images between train, validation and test sets. ➡️ [Data_splitting](../split_dataset.md)

- Load the ViT Model, explain briefly the architecture and define the metrics to add. ➡️ [VitModel_Metrics](../vit_model.md)

- Search for the best model with The Bayesian Optimization strategy.

- Fine-tune the best model.

- Evaluate the model on the test set.

- Deploy the model to Hugging Face.
