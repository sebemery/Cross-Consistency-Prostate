import train_single
import train_multi
import argparse
import json
import torch

def main(config, resume1, resume2, resume3, resume4, resume5, resume6, sites):
    if sites == 'All':
        train_multi.main(config, resume1, resume2, resume3, resume4, resume5, resume6)
    else:
        train_single.main(config, resume1, sites)



if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-r1', '--resume1', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r2', '--resume2', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r3', '--resume3', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r4', '--resume4', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r5', '--resume5', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r6', '--resume6', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('-s', '--sites', default='All', type=str,
                        help='sites to download')

    args = parser.parse_args()

    config = json.load(open(args.config))
    # for performance on gpu when input size does not vary
    torch.backends.cudnn.benchmark = True
    main(config, args.resume1, args.resume2, args.resume3, args.resume4, args.resume5, args.resume6, args.sites)