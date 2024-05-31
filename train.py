import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import DFlow
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model = DFlow(auxiliary_flow_dim=h.auxiliary_flow_dim,
                  auxiliary_layers=h.auxiliary_layers,
                  primary_flow_dim=h.primary_flow_dim,
                  primary_flow_layers=h.primary_flow_layers,
                  decoder_dim=h.decoder_dim,
                  decoder_layers=h.decoder_layers,
                  condition_channel=h.num_mels).to(device)

    if rank == 0:
        print(model)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    cp_model = None
    if os.path.isdir(a.checkpoint_path):
        cp_model = scan_checkpoint(a.checkpoint_path, 'dflow_')

    steps = 0
    if cp_model is None:
        state_dict_model = None
        last_epoch = -1
    else:
        state_dict_model = load_checkpoint(cp_model, device)
        model.load_state_dict(state_dict_model['model'])
        steps = state_dict_model['steps'] + 1
        last_epoch = state_dict_model['epoch']

    FP16 = h.FP16
    scaler = GradScaler(enabled=FP16)

    if h.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True).to(device)

    optim_model = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if cp_model is not None:
        optim_model.load_state_dict(state_dict_model['optim'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_model, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, device=device,
                          base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
                              device=device, base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    model.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            if steps < 300001:
                beta = 0.01
            else:
                beta = 0.002

            c, x, _ = batch
            x = x.unsqueeze(1)
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            c = torch.autograd.Variable(c.to(device, non_blocking=True))

            noisy_x = torch.randn_like(x) * beta + x
            with autocast(enabled=FP16):
                z_p, logs_all, x_hat = model(noisy_x, c)
                loss_ll = torch.mean(z_p ** 2) / 2 - torch.mean(logs_all)
                loss_r = 1 / beta * F.l1_loss(x_hat, x)

            loss_all = loss_ll + loss_r

            optim_model.zero_grad()
            scaler.scale(loss_all).backward()
            scaler.unscale_(optim_model)
            scaler.step(optim_model)
            scaler.update()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print(
                        'Steps : {:d}, Total loss : {:4.3f}, likelihood loss: {:4.3f}, reconstruction loss: {:4.3f}, '
                        's/b : {:4.3f}'.format(steps, loss_all, loss_ll, loss_r, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/dflow_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'model': (model.module if h.num_gpus > 1
                                               else model).state_dict(),
                                     'optim': optim_model.state_dict(),
                                     'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/total_loss", loss_all, steps)
                    sw.add_scalar("training/likelihood_loss", loss_ll, steps)
                    sw.add_scalar("training/reconstruction_loss", loss_r, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    model.eval()
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            c, x, _ = batch
                            if h.num_gpus > 1:
                                x_hat = model.module.infer(c.to(device), 0.8)
                            else:
                                x_hat = model.infer(c.to(device), 0.8)
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/x_{}'.format(j), x[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/spec_{}'.format(j), plot_spectrogram(c[0]), steps)

                                sw.add_audio('generated/x_hat_{}'.format(j), x_hat[0], steps, h.sampling_rate)
                                x_hat_spec = mel_spectrogram(x_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/x_hat_spec_{}'.format(j),
                                              plot_spectrogram(x_hat_spec.squeeze(0).cpu().numpy()), steps)
                    model.train()
            steps += 1
        scheduler.step()
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='ckpts')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=5000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
