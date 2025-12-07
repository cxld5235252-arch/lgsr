# Power by Xinlong Cheng 2024-09-04 13:04:06
import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext
import torch.nn.functional as F
from datapipe.datasets import create_dataset
from torch.nn.utils import clip_grad_norm_
from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop_with_tele, paired_random_crop_with_tele_mask
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

import torch
import torch.cuda.amp as amp
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = 1

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)
            torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        # from itertools import chain
        self.optimizer = torch.optim.AdamW(
                                           self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        # ===============================================================================
        # 1.扩散模型需要的Unet初始化和冻结
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            # 读取UnetModelSwin预训练权重
            if self.rank == 0:
                for _ in range(10):
                    self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")

            util_net.reload_model(model, ckpt)
            
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)  # wrap the network
        else:
            self.model = model

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }

        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None

        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 0),
                        pin_memory=True,
                        prefetch_factor=None,
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])

        for ii in range(self.iters_start, self.configs.train.iterations):
      
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)
            ii += 1

        # close the tensorboard
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            unet_ckpt_path = self.ckpt_dir /'unet_model.pth'
            torch.save(self.model.state_dict(), unet_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8,show=False):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging

        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=False, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            # ============================================
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            im_np = im_np*imagenet_std+imagenet_mean
            # ============================================
            # util_image.imwrite(im_np, im_path)
            import cv2
            im_np = (im_np*255).astype(np.uint8)
            im_np = np.clip(im_np,0,255)
            im_np = cv2.cvtColor(im_np,cv2.COLOR_RGB2BGR)
            cv2.imwrite(im_path, im_np)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def load_model(self, model, ckpt_path=None):
        if self.rank == 0:
            self.logger.info(f'Loading from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
        if self.rank == 0:
            self.logger.info('Loaded Done')

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        # if self.configs.train.lr_schedule == 'cosin':
        #     self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #             optimizer=self.optimizer,
        #             T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
        #             eta_min=self.configs.train.lr_min,
        #             )

    def build_model(self):
        # 初始化扩散模型
        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)
        super().build_model()

        # autoencode是冻住的
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            autoencoder.load_state_dict(ckpt, True)
            for params in autoencoder.parameters():
                params.requires_grad_(False)
            autoencoder.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        # autoencode是冻住的
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            autoencoder.load_state_dict(ckpt, True)
            for params in autoencoder.parameters():
                params.requires_grad_(False)
            autoencoder.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        # LPIPS metric
        lpips_loss = lpips.LPIPS(net='vgg').to(f"cuda:{self.rank}")
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss

        # LPIPS metric
        lpips_loss = lpips.LPIPS(net='vgg').to(f"cuda:{self.rank}")
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'realesrganmask'
        if realesrgan and phase == 'train':
            im_gt = data['gt'].cuda()
            im_tele = data['tele'].cuda()
            im_lq = data['lq'].cuda()
            im_mask = data['mask'].cuda()

            # random crop
            gt_size = self.configs.degradation['gt_size']
            # im_gt, im_lq, im_tele = paired_random_crop_with_tele(im_gt, im_lq, im_tele, gt_size, 1)
            im_gt, im_lq, im_tele, im_mask = paired_random_crop_with_tele_mask(im_gt, im_lq, im_tele, im_mask, gt_size, 1)

            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_tele = (im_tele - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            self.lq, self.tele, flag_nan = replace_nan_in_batch(im_lq, im_tele)
            self.lq, self.mask, flag_nan = replace_nan_in_batch(im_lq, im_mask)
 
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            # self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
            # return {'lq':self.lq, 'gt':self.gt, 'tele': self.tele, 'mask':self.mask}
            return {'lq':self.lq, 'gt':self.gt, 'tele': self.tele, 'mask':self.mask}
        else:
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            loss = losses['loss'].mean() / num_grad_accumulate
            loss.requires_grad_(True)
            # print(loss)
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)

            # [bz]
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
                    )

            if 'autoencoder_custom' in self.configs:
                noise_chn = self.configs.autoencoder_custom.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
                
            # [bz, noise_chn, latent_h, latent_w]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (self.configs.model.params.image_size, ) * 2,
                    device=micro_data['gt'].device,
                    )
            
            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq':micro_data['lq'],'tele': micro_data['tele']}
            else:
                model_kwargs = None
          
            # models.respace.SpacedDiffusion
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                micro_data['mask'],
                # first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs
            )

            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
           
            # make logging
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach())
        
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:   
            if self.current_iters % self.configs.train.log_freq[0] == 0:
                num_timesteps = self.base_diffusion.num_timesteps
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                            self.current_iters,
                            self.configs.train.iterations)
                for timestep in range(num_timesteps):
                    log_str += 't({:d}):{:.1e},'.format(
                                timestep,
                                loss[f'mse_{timestep}'].item()
                                )
                self.logger.info(log_str)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        None,
                        self.autoencoder
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                self.logging_image(z0_pred, tag='x0-pred', phase=phase, add_global_step=True)


class TrainerDifIRLPIPS(TrainerDifIR):
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        # diffusion loss
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                    None
                    ) # f16
            
            self.current_x0_pred = x0_pred.detach()

            # classification loss
            losses["lpips"] = self.lpips_loss(
                    x0_pred.clamp(-1.0,1.0),
                    micro_data['gt'],
                    ).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)

            losses["mse"] *= loss_coef[0]
            losses["lpips"] *= loss_coef[1]

            assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:   
            if self.current_iters % self.configs.train.log_freq[0] == 0:
                num_timesteps = self.base_diffusion.num_timesteps
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                            self.current_iters,
                            self.configs.train.iterations)
                for timestep in range(num_timesteps):
                    log_str += 't({:d}):{:.1e},'.format(
                                timestep,
                                loss[f'mse_{timestep}'].item()
                                )
                self.logger.info(log_str)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        None,
                        self.autoencoder
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


