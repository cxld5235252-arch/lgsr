# ReWiTe: Realistic Wide-angle and Telephoto Dual Camera Fusion Dataset and Loosely Alignement Based Fusion

## Environment Setup

To set up your environment, follow these steps:

```
conda create -n my_env python=3.8 -y
conda activate my_env
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -y
pip install -r requirements.txt
```

## Test our LAFusion network

Run the following command to test our LAFusion network. Results are saved in the `[result_dir]` folder.
```
CUDA_VISIBLE_DEVICES=0 python inference_LAFusion.py -i ./data/wide_tile -o [result_dir]
```

## Merge tile to full result
'''
python /data/cxl/cxl_oppo/github/LGSR/merge_tile.py
'''

## Visualize the result 

