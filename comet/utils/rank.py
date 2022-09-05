import torch

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank_0(args, message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            with open(args.log_file, "a") as f:
                f.write(message + "\n")
                f.flush()
    else:
        with open(args.log_file, "a") as f:
            f.write(message + "\n")
            f.flush()

