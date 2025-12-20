# ReWiTe: Realistic Wide-angle and Telephoto Dual Camera Fusion Dataset and Loosely Alignement Based Fusion

## Environment Setup

To set up your environment, follow these steps:

```
conda create -n my_env python=3.8 -y
conda activate my_env
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -y
pip install -r requirements.txt
```

## Visualize Input W and Input T
<p align="center">
  <img src="https://raw.githubusercontent.com/cxld5235252-arch/LAFusion/main/result/284_wide.png" alt="图1" width="45%" style="display:inline-block; margin-right:10px;">
  <img src="https://raw.githubusercontent.com/cxld5235252-arch/LAFusion/main/result/284_tele.png" alt="图2" width="45%" style="display:inline-block;">
</p>

<p align="center"><b>图 1：左为广角图，右为长焦图</b></p>



## Test our LAFusion network

Run the following command to test our LAFusion network. Results are saved in the `[result_dir]` folder.
```
CUDA_VISIBLE_DEVICES=0 python inference_LAFusion.py -i ./data/wide_tile -o [result_dir]
```

## Merge tile to full result
```
python /data/cxl/cxl_oppo/github/LGSR/merge_tile.py
```

## Visualize the result 
![Fusion Result](https://github.com/cxld5235252-arch/LAFusion/blob/main/result/284_wide.png)


