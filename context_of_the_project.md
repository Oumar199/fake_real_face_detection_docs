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
<!-- <i style="color: darkorange">
[bit_left_eye, bit_right_eye, bit_nose, bit_mouth]
</i> -->

The segment is modified if it is the positive bit (1). Otherwise, the segment is not modified. 

