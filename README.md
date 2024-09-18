# FPGANDA Official Code Description
This is the official source code of the paper 'Features kept generative adversarial network data augmentation strategy for hyperspectral image classification'

The code first trains the GAN model, then generates new data and incorporates it into the classification network for training.

Paper link: [https://www.sciencedirect.com/science/article/abs/pii/S0031320323003990]

## Installation
A standard torch environment is required, and you can follow the DeepHyperX configuration.

Reference link: [https://github.com/nshaud/DeepHyperX]

## Instructions
### Train GAN
After configuring the dataset path information, run the following code:

```
python keepGAN.py
```

### Perform Band Selection Based on Generated Results
After training the GAN and generating samples, perform band selection.

```
python Band_Select.py
```

### Train Classifier Based on Augmented Data
Based on the selected bands, merge and generate new augmented data to train the network. For detailed parameter configurations, refer to the Completed_Band_Select.py file.

```
python Completed_Band_Select.py
```


## Notes
This project is mainly developed based on open-source codes such as WGAN, WGAN-GP, CGAN, etc. The baseline and classification networks are based on the large-scale public DeepHyperX code. The dataset reading method is also the same as in DeepHyperX.

