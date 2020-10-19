# JMLN: Joint Metric Learning Network for 2D
Sketch-based 3D Shape Retrieval

The codes have been tested successfully on Ubuntu 18.04.3 LTS, Cuda 10.0, GTX 1080 Ti,  Pytorch 1.4.0, Docker images ufoym/deepo:all-py36-cu100.

Note: Docker is suggested to used.

## Update
## Data and Pre-trained Models
**Training and Test Data**

Please put related dataset into ```../dataset/```




## Usage
### Training and Test

```
# training 
sh train_all.sh

# Test after training
sh extract_all.sh
```





## Online Result
Online result for baseline is shown as below. This result is better than the one in leaderboard.
```
"score": 60.14482022065204, "top1_acc": 0.5750211582638133, "mean_f_score": 65.05269981001194
```


