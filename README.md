# Bypass-Detection-On-Synthetic-Images
Spoofing detectors for synthetic images through applying noise to synthetic images by neural networks.
You can download the derector in [](https://github.com/ZhendongWang6/DIRE).

# Updata
**2023.9.7**
A new method is applied: the gradient direction is estimated by the weighted sum of multiple trial noises. Now, even without the gradient information from the detector, we can achieve even better results. Recognition rate from 64% to 19%, at the same time without any loss of image quality.

## When the gradient information from the detector is given
We achieve a decrease in detector recognition rate from 64% to 20% while preserving the image quality to the greatest extent.

## When the gradient information is not given
The method of applying random noise multiple times is used to test the detector, so that the gradient information of the detector is not needed.

### Before processing
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/before.jpg)

### After processing (with gradient information)
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/grad.jpg)

### After processing (without gradient information)
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/no_grad.jpg)


