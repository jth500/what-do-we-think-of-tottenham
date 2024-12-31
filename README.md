# what-do-we-think-of-tottenham

Football team logo classification with a vanilla 5-layer convolutional neural network. Achieves 90% accuracy on perturbed test dataset.

## Data
The starting dataset is 100 logos for teams in Europe's top 5 leagues, taken from Alex Teboul on [Kaggle](https://www.kaggle.com/datasets/alexteboul/top-5-football-leagues-club-logos). The data are then augmented by perturbing the images, producing a dataset of 2400 images. To reproduce, move input folder to /data, and then run scripts/preprocess_imgs.py. This produces the following folder structure

```
ac-milan/
    ac-milan_original.png  
    ac-milan_aug_1.png    
    ac-milan_aug_2.png
    ...
arsenal/
    arsenal_original.png
    arsenal_aug_1.png
    arsenal_aug_2.png
    ...
```

### Example of data pertubation
![original](https://github.com/jth500/what-do-we-think-of-tottenham/blob/main/arsenal_original.png) ![pert 1](https://github.com/jth500/what-do-we-think-of-tottenham/blob/main/arsenal_aug_12.png) ![pert 3](https://github.com/jth500/what-do-we-think-of-tottenham/blob/main/arsenal_aug_17.png) ![pert 4](https://github.com/jth500/what-do-we-think-of-tottenham/blob/main/arsenal_aug_3.png)



## Model
The first model is a simple 5-layer feed-forward CNN. It achieves 90% accuracy on the test set. See notebooks/train_simple.ipynb
