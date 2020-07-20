import argparse
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from core.model import OpenNet
from dataset.data_loader import data_loader
from core.config import Config
from core.workflow import run_validation, run_test, create_runtime_opts, save_model, update_loss_scale


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/openfew/default.yaml', type=str)
parser.add_argument('--test', default=False, action='store_true')
main_opts = parser.parse_args()

opts = Config(main_opts.cfg)
opts.setup(test=main_opts.test)
opts.ctrl.cfg = main_opts.cfg
opts.print_args()

with torch.cuda.device(int(opts.ctrl.gpu_id)):
    if main_opts.test:
        opts.logger('[Testing starts] ...\n')

        opts_test = create_runtime_opts(opts, 'test')
        opts.logger('Preparing dataset: {:s} ...'.format(opts.data.name))
        test_db = data_loader(opts, opts_test, 'test')

        net = OpenNet(opts).to(opts.ctrl.device)
        state_dict_path = opts.io.best_model_file
        checkpoints = torch.load(state_dict_path, map_location='cuda:0')
        opts.logger('loading check points from {}'.format(state_dict_path))
        net.load_state_dict(checkpoints['model'], strict=False)

        run_test(opts, test_db, net, opts_test)
    else:
        net = OpenNet(opts).to(opts.ctrl.device)

        # optimizer and lr_scheduler
        optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=opts.train.lr_scheduler, gamma=opts.train.lr_gamma)

        opts.logger('[Training starts] ...\n')

        total_ep = opts.train.nep
        total_episode = total_ep * opts.fsl.iterations

        opts_train = create_runtime_opts(opts, 'train')
        opts_val = create_runtime_opts(opts, 'val')

        loss_episode = 0
        total_loss = 0.0
        for epoch in range(total_ep):

            # DATA
            opts.logger('Preparing dataset: {:s} ...'.format(opts.data.name))
            train_db = data_loader(opts, opts_train, 'train')
            val_db = data_loader(opts, opts_val, 'val')

            # adjust learning rate
            old_lr = optimizer.param_groups[0]['lr']
            if epoch == 0:
                opts.logger('Start lr is {:.8f}, at epoch {}\n'.format(old_lr, epoch))
            scheduler.step(epoch)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                opts.logger('LR changes from {:.8f} to {:.8f} at episode {:d}\n'.format(old_lr, new_lr, epoch*opts.fsl.iterations))

            for step, batch in enumerate(train_db):

                episode = epoch*opts.fsl.iterations + step

                # adjust loss scale
                update_loss_scale(opts, episode)

                loss = net(batch, opts_train, True)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                if opts.train.clip_grad:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

                # SHOW TRAIN LOSS
                loss_episode += 1
                if episode % opts.ctrl.ep_vis_loss == 0 or episode == total_episode - 1:
                    opts.logger(' [ep {:04d} ({})] loss: {:.4f}'.format(episode, total_episode, total_loss/loss_episode))
                    loss_episode = 0
                    total_loss = 0.0

                # SAVE MODEL
                if episode % opts.ctrl.ep_save == 0 or episode == total_episode - 1:
                    save_file = opts.io.model_file.format(episode)
                    save_model(opts, net, optimizer, scheduler, episode, save_file)
                    opts.logger('\tModel saved to: {}, at [episode {}]\n'.format(save_file, episode))

                # VALIDATION and SAVE BEST MODEL
                if episode % opts.ctrl.ep_val == 0 or episode == total_episode - 1:
                    if run_validation(opts, val_db, net, episode, opts_val):
                        save_file = opts.io.best_model_file
                        save_model(opts, net, optimizer, scheduler, episode, save_file)
                        opts.logger('\tBest model saved to: {}, at [episode {}]\n'.format(save_file, episode))

        opts.logger('')
        opts.logger('Training done!')
