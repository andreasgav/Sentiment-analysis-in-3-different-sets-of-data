# Sentiment-analysis-in-3-different-sets-of-data
A Convolutional Neural Network tha was trained in 3 different sets of data, to successfully identify the emotion aroused by the sight of an artwork.

This is a Python script that provides the pipeline that is required to make series of tests in order to find the best Convolutional Neural Network classifier in a sentiment analysis project. The code is written in Pythton and is based on Tensorflow. In this work Sentiment and Art Class Dataset is used, abbreviated as the _SeAC dataset_ ([link](https://github.com/andreasgav/SeAC)).

In this example Inception-V3 model is used, in order to be adopted to the available database. Inception-V3 is among the best avaialble pre-trained models for Image Classification. In the example that is provided, the model is able of achieving an accuracy rate of approximately 40+% in the first two datasets and above 55% in the third dataset. These performances refer to the unknown sample of images, set to 10% of the toal database.

It should be noted that the code in this repository is a part of a project where a big number of different network architectures, were examined in order to find the optimal model for sentiment identification provoked by works of art, so the reader should bear in mind that this script may best work as a part of extensive tests. 

## Citation

Although the use of the available resources is free, a citation to the creators of the dataset is considered necessary. So if this coding examople is used as a whole or as a part of a project, we would encourage to reference to:

**[1]** Gavros, A., Demetriadis, S. and Tefas, A. _Deep Learning in Artworks: From Art Genre Classification to Sentiment Analysis._ (2022)
