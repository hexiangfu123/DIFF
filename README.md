# DIFF

## Setup

### Requirements
```Python==3.11```

```Pytorch==2.0.1```

```transformers==4.34.1```

```scikit-learn==1.3.2```

```tqdm==4.65.0```

```fire==0.5.0```

### Dataset
We validate the effectiveness of our model in the real world dataset [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

___
## Training
data process
```
python review_pad_load_Amazon4_ratingMatrix.py Instant_Video_5.json
```

model training
```
python main.py train --dataset='Instant_Video_data' --gpu_id=2 --batch_size=8 --num_epochs=15
```

