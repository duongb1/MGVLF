# utils/misc.py
# Copyright (c) Facebook, Inc.
"""
Misc functions, including distributed helpers.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
import torchvision  # still used for _is_tracing guard


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window/global avg."""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(float(value))
        self.count += n
        self.total += float(value) * n

    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque!"""
        if not is_dist_avail_and_initialized():
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Returns list[data] from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialize to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=device) for _ in size_list]
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, t in zip(size_list, tensor_list):
        buf = t.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buf))
    return data_list


def reduce_dict(input_dict, average=True):
    """Reduce values in dict from all processes so everyone has the same averaged results."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names, values = [], []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced = {k: v for k, v in zip(names, values)}
    return reduced


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join([f"{n}: {m}" for n, m in self.meters.items()])

    def synchronize_between_processes(self):
        for m in self.meters.values():
            m.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = '' if header is None else header
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}', 'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / max(len(iterable),1):.4f} s / it)')


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))
    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha, diff, branch = 'N/A', "clean", 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    return f"sha: {sha}, status: {diff}, branch: {branch}"


def collate_fn(batch):
    # Không dùng trong pipeline hiện tại, giữ nguyên để tương thích nếu cần.
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sub in the_list[1:]:
        for i, item in enumerate(sub):
            maxes[i] = max(maxes[i], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask  # (B,H,W) bool, True = padding

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    Pad a list of (C,H,W) to a batch (B,C,H_max,W_max) + mask (B,H_max,W_max).
    Mask semantics: True = padding (khớp DETR/PE).
    """
    if tensor_list[0].ndim != 3:
        raise ValueError('Expect list of (C,H,W) tensors')
    if torchvision._is_tracing():
        return _onnx_nested_tensor_from_tensor_list(tensor_list)

    max_size = _max_by_axis([list(img.shape) for img in tensor_list])  # (C,H,W)
    b, c, h, w = [len(tensor_list)] + max_size
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    tensor = torch.zeros([b, c, h, w], dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)  # True = pad
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], : img.shape[2]] = False
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs, padded_masks = [], []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = F.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = F.pad(m, (0, padding[2], 0, padding[1]), value=1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """Disable printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Compute precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Wrap nn.functional.interpolate with the same signature.
    Keeps support for empty batch sizes implicitly handled by PyTorch.
    """
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


if __name__ == '__main__':
    samples = torch.zeros((2, 3, 64, 64))
    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list([samples[0], samples[1]])
    print('OK')
