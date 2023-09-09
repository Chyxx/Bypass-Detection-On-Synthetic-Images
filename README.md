# Bypass-Detection-On-Synthetic-Images
Spoofing detectors for synthetic images through applying noise to synthetic images by neural networks.
The detector: [see here](https://github.com/ZhendongWang6/DIRE).

# Update
**_2023.9.9_**  
Added v2 version with more mathematical rigor and comments.

**_2023.9.9_**  
1. The previous model works so well because it exploits a flaw in the detector: if you change the border color of the image, the detector doesn't work properly. However, once the border of the processed image is cropped, the recognition rate is not reduced at all. Now, by changing the distribution of the trial noise (from a Poisson distribution to a Gaussian distribution), we have managed to get the processor to work without exploiting the detector bug, although the reduction in recognition rate is not as high as before.
2. The interference noise is eliminated by denoising the trial noise, so that the true gradient descent direction is much more prominent than before.
3. The optimizer was changed from Adam to AdamW for better learning.(We also tried SGD+momentum, but it didn't optimize very well here.)

**_2023.9.7_**  
A new method is applied: the gradient direction is estimated by the weighted sum of multiple trial noises. Now, even without the gradient information from the detector, we can achieve even better results. Recognition rate from 64% to 19%, at the same time without any loss of image quality.  

**_2023.9.5_**  
The method of applying random noise multiple times is used to test the detector, so that the gradient information of the detector is not needed.However, the effect of this method is poor: the decrease of recognition rate is small, and the loss of image quality is too high.

# Result
## The changing curve of prob2/prob1 on the validation set
  ![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/prob2_prob1.png)  

## Before processing  
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/before.jpg)

## After processing (without gradient information from the detector)
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/after_best.png)

## The generated noise to add on images
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/best_noise.png)


