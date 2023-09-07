# Bypass-Detection-On-Synthetic-Images
Spoofing detectors for synthetic images through applying noise to synthetic images by neural networks.
The detector: [see here](https://github.com/ZhendongWang6/DIRE).

# Updata
**_2023.9.7_**  
A new method is applied: the gradient direction is estimated by the weighted sum of multiple trial noises. Now, even without the gradient information from the detector, we can achieve even better results. Recognition rate from 64% to 19%, at the same time without any loss of image quality.  

**_2023.9.5_**  
The method of applying random noise multiple times is used to test the detector, so that the gradient information of the detector is not needed.However, the effect of this method is poor: the decrease of recognition rate is small, and the loss of image quality is too high.


### Before processing
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/before.jpg)

### After processing (with gradient information)
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/grad.jpg)

### After processing (without gradient information)
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/no_grad.jpg)


