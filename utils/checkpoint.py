import os
import shutil
import torch

def save_checkpoint(state, is_best, args, filename='default'):
    os.makedirs('./saved_models', exist_ok=True)
    if filename == 'default':
        filename = f"MGVLF_batch{args.batch_size}_epoch{args.nb_epoch}_lr{args.lr:g}_seed{args.seed}"

    ckpt = f'./saved_models/{filename}_checkpoint.pth.tar'
    best = f'./saved_models/{filename}_model_best.pth.tar'
    torch.save(state, ckpt)
    if is_best:
        shutil.copy(ckpt, best)

def _strip_module(sd):
    return { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }

def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain, map_location="cpu")
        if "state_dict" in checkpoint:
            pretrained = checkpoint["state_dict"]
        elif "model" in checkpoint:
            pretrained = checkpoint["model"]
        else:
            pretrained = checkpoint

        pretrained = _strip_module(pretrained)
        model_dict = model.state_dict()

        matched = {k: v for k, v in pretrained.items()
                   if k in model_dict and model_dict[k].shape == v.shape}
        if not matched:
            msg = f"=> WARNING: no matching keys in {args.pretrain}"
            print(msg); logging.info(msg)
        else:
            model_dict.update(matched)
            model.load_state_dict(model_dict, strict=False)
            msg = f"=> loaded pretrain {args.pretrain} ({len(matched)} keys matched)"
            print(msg); logging.info(msg)

        del checkpoint
        torch.cuda.empty_cache()
    else:
        msg = f"=> no pretrained file found at '{args.pretrain}'"
        print(msg); logging.info(msg)
    return model

def load_resume(model, args, logging, optimizer=None):
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        logging.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer is not None and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"!! optimizer state mismatch: {e}")

        args.start_epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', None)
        print(f"=> loaded checkpoint (epoch {args.start_epoch}) best_metric {best_metric}")
        logging.info(f"=> loaded checkpoint (epoch {args.start_epoch}) best_metric {best_metric}")

        del checkpoint
        torch.cuda.empty_cache()
    else:
        print(f"=> no checkpoint found at '{args.resume}'")
        logging.info(f"=> no checkpoint found at '{args.resume}'")
    return model, optimizer
