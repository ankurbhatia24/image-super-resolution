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
#### CVPR 2020
1. Unpaired Image Super-Resolution using Pseudo-Supervision: [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf)
2. Learning to Have an Ear for Face Super-Resolution: [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Meishvili_Learning_to_Have_an_Ear_for_Face_Super-Resolution_CVPR_2020_paper.pdf)
3. Residual Feature Aggregation Network for Image Super-Resolution: [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Residual_Feature_Aggregation_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf)
4. Pulse: [Link](https://arxiv.org/pdf/2003.03808.pdf)
5. Correction Filter for Single Image Super-Resolution: [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Abu_Hussein_Correction_Filter_for_Single_Image_Super-Resolution_Robustifying_Off-the-Shelf_Deep_Super-Resolvers_CVPR_2020_paper.pdf)


#### 2018 - 2019 - 2017
1. Deep Back-Projection Networks For Super-Resolution [Link](https://arxiv.org/pdf/1803.02735.pdf)
2. Deep Laplacian Pyramid Networks (Link)[https://arxiv.org/pdf/1710.01992.pdf]
3. Very Deep Residual Channel Attention Networks (Link)[https://arxiv.org/pdf/1807.02758.pdf]
4. FSRNet: End-to-End Learning Face Super-Resolution with Facial Priors (2017) [Link](https://arxiv.org/pdf/1711.10703.pdf)

#### Previous works
   
    • A+: R. Timofte, V. De Smet, and L. Van Gool, “A+: Adjusted anchored neighborhood regression for fast super-resolution,” ACCV, 2014.
    • SRF: S. Schulter, C. Leistner, and H. Bischof, “Fast and accurate image upscaling with super-resolution forests,” CVPR, 2015.
    • SelfExSR: J.-B. Huang, A. Singh, and N. Ahuja, “Single image superresolution from transformed self-exemplars,” CVPR, 2015.
    • SCN: Z. Wang, D. Liu, J. Yang, W. Han, and T. Huang, “Deep networks for image super-resolution with sparse prior,” ICCV, 2015.
    • SRCNN: C. Dong, C. C. Loy, K. He, and X. Tang, “Image superresolution using deep convolutional networks,” TPAMI, 2015.
    • VDSR: J. Kim, J. K. Lee, and K. M. Lee, “Accurate image superresolution using very deep convolutional networks,” CVPR, 2016.
    • DRCN: J. Kim, J. K. Lee, and K. M. Lee, “Deeply-recursive convolutional network for image super-resolution,” CVPR, 2016.
    • FSRCNN: C. Dong, C. C. Loy, and X. Tang, “Accelerating the superresolution convolutional neural network,” ECCV, 2016.
    • DRRN: Y. Tai, J. Yang, and X. Liu, “Image Super-Resolution via Deep Recursive Residual Network,” CVPR, 2017.
