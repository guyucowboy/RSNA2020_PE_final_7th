# RSNA STR Pulmonary Embolism Detection 


## 1. General

Competition Name: [RSNA STR Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detectionn)

Team Name: yuval reina

Private Leaderboard Score: 0.157

Private Leaderboard Place: 7

Members:



*   Yuval Reina, Tel - Aviv, Israel, [Yuval.Reina@gmail.com](mailto:Yuval.Reina@gmail.com)


## 2. Background:


### Yuval Reina:


### Academic and professional background

I hold a BSc in Electrical Engineering and an MBA. I have worked for the last 30 years as an R&D engineer and executive manager in different areas, all related to H/W design and RF (no AI or ML). 

ML knowledge comes from online courses (Coursera, Udemy), online articles and Kaggle competitions - Kaggle GM.

**Why this competition?**

I like medical imaging competitions.


### Time spent on the competition

I spent about 7H a week on this competition, which sums up to around 70H overall. 


## 3. Summary

My solution is based on two step model + Ensemble:



1. Base model for feature extraction per CT slice
2. Transformer model - combining all the output features from a series and predict per series and per image. 
3. Ensembling is done in a special way - ‘2nd opinion ensembling’
4. Post - processing.


### Base Model:

As base model I used a models from the [EfficientNet](https://arxiv.org/abs/1905.11946) family[4]:



*   EfficientNet b3 
*   EfficientNet b5 

All models were pre-trained on imagenet using noisy student algorithms. The models and weights are from  [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)[2].

The input is converted from 1 channel high dynamic range to 3 channels with lower dynamic range using a windowing mechanism, but unlike the usual solution which uses fixed windows, I used [WSO](https://arxiv.org/pdf/1812.00572.pdf)[5] where the windows’ parameters are optimized during the model’s training.

The input to the base model is a single CT slice and the output is a prediction of the image class - ‘PE Present in Image’ and the series classes, which are set to zero if ‘PE Present in Image’ is zero.

The input to the last linear classification layer is a vector of size 256 which is the features vector and it is the next stage’s input.


### Transformer Models:

The input to the Transformer models are a stack of features from all CT slices belonging to the same series.

The transformer is a stack of 4-6 transformer encoder layers with self attention as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) [1]. Each transformer encoder layer uses 2-4 self attention heads. 

The outputs of the transformer are: 



1. Vector of size N - Number of slices - which is the prediction for ‘PE Present in Image’ per image.
2. Vector of size 9 - Number of classes per series

The instace_number - the order of the CT slice in the series is used as a positional embedding, and the relative place is also calculated and after it is expanded by  a small MLP network it is also added to the image feature vector.    


### Ensembling - A 2nd opinion 

As the test data set is very large, the ensemble is handicapped, as I could only fit  2 models in the 9 hours time limit. To solve this I ensemble only images and series with the highest uncertainty - i.e. the output is closer to 0.5. 


### Post Processing

The output predictions must adhere to the expected label hierarchy defined in the competitions rules. Although most of the predictions adhere to these rules, for some series where the model is highly uncertain, the prediction may deviate from these rules. In these cases the post - processing mechanism does the minimal changes needed for the predictions to obey the rules. 


### Training

The heavy lifting was the training and inference of the base models. This was done on a server with 2 GPUs – Tesla V100, Titan RTX that worked in parallel on different tasks. Training one fold of one model took ~ 9H for B5 models  on the Tesla and 20% more on the Titan. Inferencing for this fold took ~3H on the Tesla.

The transformer training took 10-15 minutes of the Tesla.

The total time it took to train all models used in the final submission is about and folds is about 44H one Tesla.


## 4. Models and features


### Base models

As base models I tried various types of models (pre-trained on Imagenet):



*   Densenets – 201
*   EfficientNet B3 , B4, B5, B6, B7 with and with noisy student pre-training and with normal pretraining 
*   se_resnext101_32x4d, se_resnext50_32x4d

    At the end I used EfficientNet as it was best when judging accuracy/time


    The noisy student version performed better than the normal one.


    I used the original image size of the CT’s (512*512).



### WSO [5]

The dynamic range of the images must be addressed otherwise there is a significant degradation of the score. The straightforward way is to use windowing, the same way radiologists do. As all models are designed for color images, they all have 3 channels as the input layer, hence 3 windows is a very convenient solution. Instead of using constant windows like - Lungs, PE, Bones, etc., I used adaptive windowing as I did in [my last year's solution](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117480). 

 The idea here is to add a very simple block built of 3 layers:



*   Conv2d(1, 3, kernel_size=(1, 1))
*   Sigmoid()
*   InstanceNorm2d(3)

The first layer is a simple cell which multiply every pixel by the same value and add the same bias, per channel, then comes the sigmoid that acts as the window, and the instance norm just normalizes the image. WSO slightly improved the performance.


#### Features

The final layer of the standard EfficientNet is a vector size 2048, from my past experience I know that for images which are as similar as the CT’s, this vector is sparsely occupied and most of its values are close to zero. Hance I added another layer which squeezed this vector to size 256, I tried linear layer and pooling layer and got similar results. 


#### Model’s output

As the model’s output I used:



*   PE present in image
*   Intermediate and the CT quality classes (except _true_filling_defect_not_pe_) - In some models I omitted the Intermediate class. 
*   All other series classes - these classes were set to 0 if no PE was present in the  image.


### Transformer Network

The input to the transformer network is the features from all the images from one patient.

The inspiration for this kind of model came from[ last year's RSNA competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) when all the top solutions (including my own) used a two stage network approach (although none of them were transformers).

Using a transformer seems appropriate in this case because transformers are built to seek relationships between embedding (feature) vectors in it’s input. And the ability of transformers to classify ordered series of images.


#### CT slice location

In NLP transformer models, the location of the word is embedded and added to the word embedding vector. In CT scans we can do the same, the order of the series is specified in the value of ‘InstanceNumber’  and I used it to calculate the positional embedding. I also calculated the relative place in the series by calculating (x-xmin)/(xmax-xmin) to get a continuous value which I turned into an embedding vector of size 256 using a simple 2 layers MLP.

I later found out that in some series the InstanceNumber doesn’t reflect the true position and I used the real z position to reorder the CT slices, but I didn’t see any improvement in the CV or LB.


### Padding 

As every series has a different number of slices the transformer needs to handle a different number of feature vectors. This is done by padding the end of the transformer with zero vectors. 


#### Limit the input size

A transformer can be trained on different number of feature vectors, by using padding. But when the range of numbers is very large, from just more than 60 in some cases to more than 1000 in others this may cause some implementation issues (mainly long train and inference time). To simplify these issues, I limited the input to N  feature vectors during training and inference, and for each series I randomly divided the CTs to groups of size up to N.  I tried different values of N and found out N=128 is the best.

This may also be used as a kind of TTA (Test Time Augmentation) during inference as the full series is split randomly.


### Output

The output of the standard transformer is a matrix size N*C when N is the number of input feature vectors and C=256 is the size of each feature vector. We get the outputs of the model by using 2 different linear layers on this matrix 

The outputs of the transformer are:



*   PE present in image - one output per non padded image - this is achieved by passing the N*C matrix through a C->1 linear layer and keeping only the first M images which are real CT’s and not padding vectors.
*   Series classes (_Chronic, Left PE, Right PE, _etc.) - 9 outputs per series, these are calculated by passing only the N’s output vector through a C->9 linear layer. When the input matrix is built I make sure the last input vector is always a padding vector, hence the last output vector always corresponds to a padding vector.


## 5. Training and Inferencing


### Preprocessing

I used the original size of the CTs, i.e 512*512 (as a precaution I did a simple crop-pad in case one of the CT is not 512*512).

As the DICOM image array can have very large pixel values these values were clipped to [-3000,3000].


### Loss Function

The loss function for the base model used was weighted binary cross entropy, the weights were designed to mimic the competition’s metric weights, with ‘PE present in image’ got a weight of 0.33 and the series classification classes got the other 0.67.

The loss function for the transformer was the same as the competition’s metric.

### Base model sampling

As most of the targets for the single CT slices are zeros, I tried different sampling methods. I ended up with a sampling function that emphasizes slices with positive targets or which are near the center of the series.

### Augmentation


#### Base model

The following augmentations where used while training and inference:

Random resize + crop

Random rotation

Random flip

Random mean/std pixel value change, 

[Cutout](https://arxiv.org/abs/1708.04552)[3] - erasing a small rectangle in the image


### Transformer

From the base model I extracted the feature vectors without augmentation, as the feature extraction time was very long. As augmentation I added random vectors to the features.

Another augmentation is the random grouping as stated above.


#### Inference 

As stated above, the inference was done using basic TTA as CT augmentation takes too much time. The features vectors were extracted without augmentation. While inference with the transformer model, I added random vectors to the feature vectors and used random splits to sub - series of length 128. For every transformer the inference ran 12 times.


### Ensembling - 2nd opinion method

Inferencing the full test dataset took ~4H per model on kaggle’s kernel, as the time limit was 9H I could only fit 2 full models. To gain some more ensembling I used a “2nd opinion mechanism”. The idea is not to run another model and ensemble all the series, but only the series with the highest uncertainty. The uncertainty is measured by how close the series classes to 0.5 (the closer, the most uncertain), and the same for the CT’s ‘PE present on image’.

Ensembling is then done like this:



1. The first stage was to do full inference with the best single model.
2. Then use the uncertainty to select about 50% of the series and interface with a 2nd model.
3.  Ensemble
4. Select 50% of the most uncertain series again, and inference with a 3rd model.
5. Ensemble
6. Select 25% of the most uncertain series, and inference with a 4rd model.
7. Ensemble

As the 4 models for ensembling I used:



1. base - EfficientNet b5 noisy student, fold 0/5, transformer -  4 encoders, 2 attention heads, 2048 feedforward dim.
2. base - EfficientNet b5 noisy student, fold 2/5, transformer -  4 encoders, 2 attention heads, 3072 feedforward dim.
3. base - EfficientNet b5 noisy student, fold 1/5, transformer -  4 encoders, 4 attention heads, 3072 feedforward dim.
4. base - EfficientNet b3 noisy student, fold 0/5, transformer -  6 encoders, 4 attention heads, 2048 feedforward dim.


### Post processing

The output prediction must comply with certain restrictions which are stated in [the label consistency requirements. ](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/183473)Although most predictions comply with these restrictions, some don’t. I tried to change the loss function to penalize the network for not compiling, but it didn’t really decrease the non-compliant prediction.  

We get non-compliant predictions when the network is uncertain about the prediction, i.e. if the network is certain there isn’t any PE in any of the images, all the classes will comply. But if for some images the network is uncertain about the existence of PE or which kind of PE exists, the images will get ~0.5 for ‘PE present on image” and/or some of the series classes will be near 0.5. In these cases the predictions might not be consistent.

To solve this issue, I used a heuristic to post-process the predictions when labels aren’t consistent. This heuristic tries to make the smallest change to the prediction in order to make them consistent.

Some of the heuristic rules are:



1. If _Negative for PE_>0.5 if any other series prediction > _Negative for PE then Negative for PE=0.4999 else clip all _other series prediction to 0.4999
2. If _Indeterminate_>0.5 if any other series prediction > _Indeterminate then Indeterminate=0.4999 else clip all _other series prediction to 0.4999
3. If _Indeterminate_>0.5 or _Negative for PE_>0.5 then clip _PE Present on Image  for all the images in the series _to 0.4999. Else make sure the image with the highest _PE Present on Image>=0.5001_
4. The heuristic continue in this way for all the other requirements (It isn’t really important and can be underwood by reading the code)

The LB penalty for using this heuristic was small ~0.001 or less, this is because the inconsistent predictions were near 0.5.


## 6. Interesting findings


### Sparse feature vectors

At the beginning of this competition and in previous competitions I assumed a feature vector of size 256 will give a good representation of the image with enough information. After some of my transformers over fitted too fast, I checked the feature vectors and found out most of the values are near 0, this is because the last layer isn’t really needed and as the base model is fine tuned It has enough parameters to predict the classes without the last layer and this layer become a very degraded layer. 

The solution I found to this issue was to use a drop-out layer with high P  before the last linear layer.


## 7. Better “Real World” model

One base model with one transformer is already a very simple solution which gives quite good private LB ~ 0.164 - 0.170.

I believe a more unified model for CT images can be designed by considering also the models from last year’s RSNA competition. These models were also 2 stage models and I believe a model like the one I used this year would have scored high in the last year’s competition too (after fine-tuning). The heavy lifting in both competitions was training the base models. I believe we can train a base model to ‘see’ and extract features for different kinds of CT images. This can be done by using CT slices for different body parts and let the network predict which organs are present in the CT and  any medical diagnosis. After we have a trained base network we can use it for different tasks like PE detection or Intracranial hemorrhage detection etc., for each task we only need to extract features using the base model and to fine tune the transformer to do the specific task.


## 8. Model Execution Time

All numbers refer to training on a system with intel i9-9920, 64GB RAM, Tesla V100 32G GPU.

Training base models ~ 9 h/fold

Inferencing the train data to prepare the transformer network input 3h 

Training transformer network 15min/model 

Which sums up to ~44H for the 4 models I used (1 was smaller model - b3)

Inferencing took less than 9H


## 9. References


    [1] Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin, _Attention Is All You Need_, 2017 [https://arxiv.org/abs/1706.03762v5](https://arxiv.org/abs/1706.03762v5)


    [2] Ross Wightman,  [https://github.com/rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)


    [3] Terrance DeVries and Graham W. Taylor,_ Improved Regularization of Convolutional Neural Networks with Cutout_ ,2017, [https://arxiv.org/abs/1708.04552](https://arxiv.org/abs/1708.04552)


    [4] Mingxing Tan and Quoc V. Le, _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_, 2019, [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)


    [5] Hyunkwang Lee, Myeongchan Kim, Synho Do, Practical Window Setting Optimization for Medical Image Deep Learning, 2018, [https://arxiv.org/pdf/1812.00572.pdf](https://arxiv.org/pdf/1812.00572.pdf)
