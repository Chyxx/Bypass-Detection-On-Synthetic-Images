# Bypass-Detection-On-Synthetic-Images
Spoofing detectors for synthetic images through applying noise to synthetic images by neural networks.

## When the gradient information from the detector is given
We achieve a decrease in detector recognition rate from 64% to 20% while preserving the image quality to the greatest extent.

### Before processing
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/sdv2-test.jpg?raw=true)

### After processing
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/after_process_test.jpg?raw=true)


## Without gradient information from the detector
The method of applying random noise multiple times is used to test the detector, so that the gradient information of the detector is not needed.

### Before processing
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/sdv2.jpg?raw=true)

### After processing
![](https://github.com/Chyxx/Bypass-Detection-On-Synthetic-Images/blob/main/images/after_process.jpg)

