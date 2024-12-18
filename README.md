# ViT for Monocular Depth Estimation    

Vision Transformers-relative and absolute depth estimation    
    
## Usage

1. Images can be manually transferred to [`input`](vit_for_depth_estimation/input) folder or be downloaded from DuckDuckGo API using the script:


```shell
python fetch_sample_images.py -i <search image> -u <no. of urls>
``` 

2. Select one of the four models:
    - `DPT_Large`: Largest model 
    - `DPT_Hybrid`
    - `MiDaS`
    - `MiDaS_small`
3. Inference:

```shell
python inference.py -i ../input -o ../output -t DPT_Large
```    

```shell
python inference.py -i ../input -o ../output -t DPT_Hybrid
```
```shell
python inference.py -i ../input -o ../output -t MiDaS
```

```shell
python inference.py -i ../input -o ../output -t MiDaS_small
```
4. Absolute Depth Estimation

The models perform relative depth estimation. To approximately estimate absolute depth, method prescribed in Section-5 of the [paper](https://arxiv.org/pdf/1907.01341.pdf) has been implemented using [depth-alignnment](https://github.com/S-B-Iqbal/ViT-for-Monocular-Depth-Estimation/blob/85bbc01ab50db06a9690d4a3669a01c4e7d1264f/src/utils.py#L46). Also have a look at the following issues: [#36](https://github.com/isl-org/MiDaS/issues/36), [#37](https://github.com/isl-org/MiDaS/issues/37), [#42](https://github.com/isl-org/MiDaS/issues/42), [#63](https://github.com/isl-org/MiDaS/issues/63), [#148](https://github.com/isl-org/MiDaS/issues/148), [#171](https://github.com/isl-org/MiDaS/issues/171).    

To perform absolute depth estimation, use the below script. 

```shell
python inference.py -i ../input -o ../output -t <model_name> -a true
```

4. Output

- Results are saved in [`output`](vit_for_depth_estimation/output) folder in png format. Output for any random image can be visualized using the script:

```shell
python plot.py
``` 


## NOTE:

Training script is not provided by the original authors, refer issue [#43](https://github.com/isl-org/MiDaS/issues/43). The authors utilize the strategies proposed in the paper ["Multi-Task Learning as Multi-Objective Optimization"](https://arxiv.org/abs/1810.04650) for training on different datasets with different objectives. The authors have shared the loss function in pytorch code [here](https://gist.github.com/ranftlr/1d6194db2e1dffa0a50c9b0a9549cbd2)
