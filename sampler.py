# Power by Xinlong Cheng 2024-09-04 13:04:06
import os, math, random

import numpy as np
from pathlib import Path

from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset

class BaseSampler:
    def __init__(
            self,
            configs,
            use_amp=True,
            chop_bs=1,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_amp = use_amp

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = 1

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend='nccl', init_method='env://')

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        self.freeze_model(model)
        self.model = model.eval()
        
        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class ResShiftSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False, tele_tensor=None, mask_tensor=None):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        if self.configs.model.params.cond_lq:
            model_kwargs={
                    'lq':y0
                    }
        if tele_tensor is not None:
            model_kwargs['tele'] = tele_tensor
        # if mask_tensor is not None:
        #     model_kwargs['mask'] = mask_tensor
        h,w = y0.shape[-2:]
        with torch.no_grad():
            # results = self.base_diffusion.p_sample_loop(
            #         y=y0,
            #         model=self.model,
            #         first_stage_model=self.autoencoder,
            #         noise=None,
            #         noise_repeat=noise_repeat,
            #         clip_denoised=(self.autoencoder is None),
            #         denoised_fn=None,
            #         model_kwargs=model_kwargs,
            #         progress=False,
            #         )    # This has included the decoding for latent space
            _,_,results = self.base_diffusion.training_losses(
                self.model,
                y0,y0, mask_tensor, self.autoencoder,model_kwargs
            )
            # results = self.base_diffusion.decode_first_stage(results, self.autoencoder)
        # if flag_pad:
        #     results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]
        results = F.interpolate(results, (h,w), mode='bicubic')
        return results

    def inference(self, in_path, out_path, bs=1, noise_repeat=False):
        def _process_per_image(im_lq_tensor, tele_tensor=None, mask_tele=None, mask=None):

            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            with context():
                im_sr_tensor = self.sample_func(
                        im_lq_tensor,
                        noise_repeat=noise_repeat,
                        tele_tensor=tele_tensor,
                        mask_tensor=mask
                        )     # 1 x c x h x w, [-1, 1]

            # im_sr_tensor = im_sr_tensor * 0.5 + 0.5

            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path

        if self.rank == 0:
            assert in_path.exists()
            if not out_path.exists():
                out_path.mkdir(parents=True)

        if self.num_gpus > 1:
            dist.barrier()

        if in_path.is_dir():
            data_config = {'type': 'base',
                            'params': {'dir_path': str(in_path),
                                        'need_path': True,
                                        'extra_dir_path': self.configs.data.val.params.extra_dir_path,
                                        'mask_dir_path': self.configs.data.val.params.mask_dir_path
                                        }
                            }

            dataset = create_dataset(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            for data in dataloader:
                micro_batchsize = math.ceil(bs / self.num_gpus)
                ind_start = self.rank * micro_batchsize
                ind_end = ind_start + micro_batchsize
                micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}

                if micro_data['lq'].shape[0] > 0:
                    lq = util_image.tensor2img(micro_data['lq'], rgb2bgr=True, min_max=(0.0, 1.0))
                    util_image.imwrite(lq, 'test.png', chn='bgr', dtype_in='uint8')
                    tele = util_image.tensor2img(micro_data['tele'], rgb2bgr=True, min_max=(0.0, 1.0))
                    util_image.imwrite(tele, 'tele.png', chn='bgr', dtype_in='uint8')
                    results = _process_per_image(
                            ((micro_data['lq']-0.0)/1.0).cuda(),((micro_data['tele']-0.0)/1.0).cuda(), mask=micro_data['mask'].cuda()
                            )    # b x h x w x c, [0, 1], RGB
                    for jj in range(results.shape[0]):
                        # im_np = results[jj].cpu().permute(1,2,0).numpy()
                        # imagenet_mean = np.array([0.485, 0.456, 0.406])
                        # imagenet_std = np.array([0.229, 0.224, 0.225])
                        # im_np = im_np*imagenet_std+imagenet_mean
                        # import cv2
                        # im_np = (im_np*255).astype(np.uint8)
                        # im_np = np.clip(im_np,0,255)
                        # im_np = cv2.cvtColor(im_np,cv2.COLOR_RGB2BGR)
                        # im_name = Path(micro_data['path'][jj]).stem
                        # im_path = out_path / f"{im_name}.png"
                        # cv2.imwrite(str(im_path), im_np)
                        im_sr = util_image.tensor2img(results[jj], rgb2bgr=False, min_max=(0.0, 1.0))
                        im_name = Path(micro_data['path'][jj]).stem
                        im_path = out_path / f"{im_name}.png"
                        util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
                # micro_batchsize = math.ceil(bs / self.num_gpus)
                # ind_start = self.rank * micro_batchsize
                # ind_end = ind_start + micro_batchsize
                # micro_data = {key:value[ind_start:ind_end] for key,value in data.items()}

                # if micro_data['lq'].shape[0] > 0:
                    # results = _process_per_image(
                    #         ((micro_data['lq']-0.0)/1.0).cuda(),((micro_data['tele']-0.0)/1.0).cuda(), mask=micro_data['mask'].cuda()
                    #         )    # b x h x w x c, [0, 1], RGB
   
                    # for jj in range(results.shape[0]):
    
                        # im_np = results[jj].cpu().permute(1,2,0).numpy()
                        # imagenet_mean = np.array([0.485, 0.456, 0.406])
                        # imagenet_std = np.array([0.229, 0.224, 0.225])
                        # im_np = im_np*imagenet_std+imagenet_mean
                        # import cv2
                        # im_np = (im_np*255).astype(np.uint8)
                        # im_np = np.clip(im_np,0,255)
                        # im_np = cv2.cvtColor(im_np,cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(out_path, im_np)
            if self.num_gpus > 1:
                dist.barrier()

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

if __name__ == '__main__':
    pass

