# Plain-`python` launcher for MAE pretraining (no torchrun).
#
# Mirrors the I-JEPA launcher that works on this cluster: it spawns one process
# per GPU, isolates each process to a SINGLE physical GPU via CUDA_VISIBLE_DEVICES
# (so each rank sees only its own card as cuda:0), and sets up the NCCL process
# group over localhost. This avoids torchrun's "all ranks see all GPUs" model,
# which on this node left every rank with a stray context on GPU 0.
#
# Usage (note: still keep NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1 from gmae_rc on
# the L40S node):
#
#   . /home/t006d/gmae_rc && python main_pretrain_spawn.py \
#       --data_path ... --output_dir ... --model mae_vit_large_patch16 \
#       --batch_size 64 --epochs 800 --warmup_epochs 40 \
#       --mask_ratio 0.75 --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05 \
#       --num_workers 16 --pin_mem --compile
#
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from main_pretrain import get_args_parser, main


def _worker(rank, args, gpu_ids, world_size, master_addr, master_port):
    # Each process sees exactly ONE physical GPU (its own) -> it becomes cuda:0.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids[rank]
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = '0'  # only one device is visible to this process
    main(args)


def run():
    parser = get_args_parser()
    parser.add_argument('--ngpus', type=int, default=None,
                        help='number of processes/GPUs to spawn (default: all visible)')
    parser.add_argument('--dist_port', type=int, default=29500,
                        help='TCP port for the localhost NCCL rendezvous')
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Respect an existing CUDA_VISIBLE_DEVICES (e.g. set by LSF); else range(ngpus).
    # These are the PHYSICAL GPU ids we hand out one-per-process.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible:
        gpu_ids = [d.strip() for d in visible.split(',') if d.strip() != '']
    else:
        n = args.ngpus or torch.cuda.device_count()
        gpu_ids = [str(i) for i in range(n)]
    if args.ngpus is not None:
        gpu_ids = gpu_ids[:args.ngpus]

    world_size = len(gpu_ids)
    assert world_size >= 1, 'no GPUs available to spawn on'

    print(f'[spawn] launching {world_size} process(es) on physical GPU(s) {gpu_ids}; '
          f'rendezvous 127.0.0.1:{args.dist_port}', flush=True)

    # spawn (not fork): each child is a fresh process, so setting CUDA_VISIBLE_DEVICES
    # inside it takes effect before CUDA is initialized.
    mp.spawn(
        _worker,
        args=(args, gpu_ids, world_size, '127.0.0.1', args.dist_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == '__main__':
    run()
