# image-super-resolution
Reconstructing a high-resolution photo-realistic image from its counterpart low-resolution image.  
To date,mainstream algorithms of SISR are mainly divided into threecategories:  
1. interpolation-based  methods,  
2. reconstruction-basedmethods and 
3. learning-based methods.

Bear in mind that none of these objective measures are particularly good at predicting human visual response to image quality. Sometimes PSNRs vary wildly between two almost indistinguishable images; similarly you can have two images with the same PSNR where there is a very obvious difference in quality. The structural similarity index measurement (SSIM) and some of its variations are generally considered better from this perspective, but still not perfect models for human perception.

## Problem of Some Conventional SR Approaches
Convolutional neural nework (CNN) approaches such as SRCNN, FSRCNN and VDSR

    Firstly upscale/upsample the LR image
    Then perform convolution to get the HR images
Since the LR image is upsampled at the very beginning, all the convolutions will be based on the upsampled LR image. Thereby, the number of computation is increased.

## ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
[]()


## Resources:
Espcn in keras: [https://github.com/jlfilho/ESPCN-Keras](https://github.com/jlfilho/ESPCN-Keras) <br>
Various Superresolution models in keras: [https://github.com/titu1994/Image-Super-Resolution](https://github.com/titu1994/Image-Super-Resolution)

## References

    • A+: R. Timofte, V. De Smet, and L. Van Gool, “A+: Adjusted anchored neighborhood regression for fast super-resolution,” ACCV, 2014.
    • SRF: S. Schulter, C. Leistner, and H. Bischof, “Fast and accurate image upscaling with super-resolution forests,” CVPR, 2015.
    • SelfExSR: J.-B. Huang, A. Singh, and N. Ahuja, “Single image superresolution from transformed self-exemplars,” CVPR, 2015.
    • SCN: Z. Wang, D. Liu, J. Yang, W. Han, and T. Huang, “Deep networks for image super-resolution with sparse prior,” ICCV, 2015.
    • SRCNN: C. Dong, C. C. Loy, K. He, and X. Tang, “Image superresolution using deep convolutional networks,” TPAMI, 2015.
    • VDSR: J. Kim, J. K. Lee, and K. M. Lee, “Accurate image superresolution using very deep convolutional networks,” CVPR, 2016.
    • DRCN: J. Kim, J. K. Lee, and K. M. Lee, “Deeply-recursive convolutional network for image super-resolution,” CVPR, 2016.
    • FSRCNN: C. Dong, C. C. Loy, and X. Tang, “Accelerating the superresolution convolutional neural network,” ECCV, 2016.
    • DRRN: Y. Tai, J. Yang, and X. Liu, “Image Super-Resolution via Deep Recursive Residual Network,” CVPR, 2017.
