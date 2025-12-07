训练用main.py，CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path ./configs/ref_sr.yaml
测试用inference_resshift.py。 python inference_resshift.py -i [wide_dir] -o [result_dir]
编解码和diffusion相关的东西主要在/models/gaussian_diffusion.py

数据位置：
gt_root: /home/zkyd/cxl/code/paper/Resshift_0902/data_new_gt_0925/train/gt_y_128
lq_root: /home/zkyd/cxl/code/paper/Resshift_0902/data_new_gt_0925/train/wide_y_128
tele_root: /home/zkyd/cxl/code/paper/Resshift_0902/data_new_gt_0925/train/ref_y_128

