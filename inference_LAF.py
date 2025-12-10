
# Power by Xinlong Cheng 2024-09-04 13:04:06
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from basicsr.utils.download_util import load_file_from_url


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
            "--task",
            type=str,
            default="ref_sr",
            choices=['ref_sr'],
            help="Chopping forward.",
            )
    args = parser.parse_args()

    return args

def get_configs(args):
    if args.task == 'ref_sr':
        configs = OmegaConf.load('./configs/ref_sr_rewite_mask.yaml')
 
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            chop_bs=1,
            use_amp=True,
            seed=args.seed
            )

    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
