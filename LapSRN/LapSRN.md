# LapSRN

![LapImgae](https://imgur.com/a/ts6ZGcS)

## Architecture

Insert the image of lapsrn here

The neural net directly interpolates and enlarges the image and to reconstruct it. We use a convolutional layer and an upsampling filter to enlarge layer by layer. This structure can perceive more information.

The network progressively reconstructs the sub-band residuals of high-resolution images at multiple pyramid levels, specifically at log2(S) levels where S is the scale factor (i.e., 2, 4, 8). This notebook is specifically for 4x resolution.

The model is based on the Laplacian pyramid structure, and the input is a low-resolution image, not an interpolated and filled image. The model consists of two parts, feature extraction and image reconstruction.

### Feature extraction - 

This establishes a nonlinear map through convolution and then upsample to obtain a graph. While during reconstruction, we first perform upsampling, and then directly add the feature extraction map to this map, which is equivalent to the fusion of the two maps.

Feature extraction involves 5 layers of convolutions with a leaky relu as non-linear function where the first layer has 1 as the in_channels (to account for the grayscale) and the further down convolutions have 64 filters. After the convolutions, we have a transposed convolution at the end.

### Image reconstruction - 

The images are upsampled from the lower resolution i.e. from the LR image to 2x or from 2x to 4x by the use of upsampling kernel kernel. The input image is upsampled by a scale of 2 with a transposed convolutional (upsampling) layer. As mentioned this layer is initialized with the bilinear kernel.

A transposed convolution on the features extracted from the previous resolution is then combined (using element-wise summation) with the predicted residual image from the feature extraction branch to produce a high-resolution output image.

### How does LapGAN differ from this one?

LAPGAN is upsampled and then passed down after convolution unlike the use of low-resolution images as inputs here. Also the use case of LapGANs are different.

### Details of the model

64 filters in all convolutional layers except the first layer which applied on input image, the layers for predicting the residuals, and the image upsampling layer.

Filter size of the convolutional and transposed convolutional layers are 3×3 and 4×4 respectively.

Leaky ReLUs with slope of 0.2 is used.

Batch size of 64 and the size of HR patches is cropped as 128×128 for the paper however, I found better results for 256x256 images where the images being super resolved till 4x.

Data augmentation: (1) Scaling by randomly downscale images between [0.5, 1.0] in the paper, again better results found with scaling from [0.9, 1] perceptually (2) Random rotation of 90, 180 or 270 degrees in the paper. Found that perceptually much better results are achieved without the rotations although loss functions don't convey much details about that  (3) Random horizontally flipping with probability of 0.5.