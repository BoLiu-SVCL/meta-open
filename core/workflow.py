import torch

from core.general_utils import AttrDict


def roc_area_calc(dist, closed, descending, total_height, total_width):
    _, p = dist.sort(descending=descending)
    closed_p = closed[p]

    height = 0.0
    width = 0.0
    area = 0.0
    pre = 0  # (0: width; 1: height)

    for i in range(len(closed_p)):
        if closed_p[i] == -1:
            if pre == 0:
                area += height * width
                width = 0.0
                height += 1.0
                pre = 1
            else:
                height += 1.0
        else:
            pre = 0
            width += 1.0
    if pre == 0:
        area += height * width

    area = area / total_height / total_width
    return area


def update_loss_scale(opts, episode):
    for i in range(len(opts.train.loss_scale_entropy_lut[0])):
        if episode < opts.train.loss_scale_entropy_lut[0][i]:
            opts.train.loss_scale_entropy = opts.train.loss_scale_entropy_lut[1][i]
            break
    for i in range(len(opts.train.loss_scale_aux_lut[0])):
        if episode < opts.train.loss_scale_aux_lut[0][i]:
            opts.train.loss_scale_aux = opts.train.loss_scale_aux_lut[1][i]
            break


def save_model(opts, net, optimizer, scheduler, episode, save_file):
    file_to_save = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'opts': opts,
        'episode': episode
    }
    torch.save(file_to_save, save_file)


def create_runtime_opts(opts, split):
    if split == 'train':
        opts_runtime = AttrDict()
        opts_runtime.n_way = opts.fsl.n_way
        opts_runtime.k_shot = opts.fsl.k_shot
        opts_runtime.m_query = opts.fsl.m_query
        opts_runtime.p_base = opts.fsl.p_base
        opts_runtime.iterations = opts.fsl.iterations
        opts_runtime.open_cls = opts.open.n_cls
        opts_runtime.open_sample = opts.open.m_sample
        opts_runtime.aug_scale = 1
        opts_runtime.fold = 1
    elif split == 'val':
        opts_runtime = AttrDict()
        opts_runtime.n_way = opts.fsl.n_way_val
        opts_runtime.k_shot = opts.fsl.k_shot_val
        opts_runtime.m_query = opts.fsl.m_query_val
        opts_runtime.p_base = 0
        opts_runtime.iterations = opts.fsl.iterations_val
        opts_runtime.open_cls = opts.open.n_cls_val
        opts_runtime.open_sample = opts.open.m_sample_val
        opts_runtime.aug_scale = 1
        opts_runtime.fold = 1
    else:  # split == 'test'
        opts_runtime = AttrDict()
        opts_runtime.n_way = opts.fsl.n_way_test
        opts_runtime.k_shot = opts.fsl.k_shot_test
        opts_runtime.m_query = opts.fsl.m_query_test
        opts_runtime.p_base = 0
        opts_runtime.iterations = opts.fsl.iterations_test
        opts_runtime.open_cls = opts.open.n_cls_test
        opts_runtime.open_sample = opts.open.m_sample_test
        opts_runtime.aug_scale = opts.model.aug_scale_test
        opts_runtime.fold = opts.model.fold_test

    return opts_runtime


def evaluation(net, input_db, mode, opts_eval):

    net.eval()
    with torch.no_grad():
        total_counts = 0
        total_correct = 0.0

        pred_closed_list = []
        target_list = []
        open_score_list = []
        closed_list = []
        for j, batch_test in enumerate(input_db):

            if mode.startswith('open'):
                pred_closed, target, open_score, closed = net(batch_test, opts_eval, False)
                pred_closed_list.append(pred_closed)
                target_list.append(target)
                open_score_list.append(open_score)
                closed_list.append(closed)
            elif mode == 'regular':
                correct = net(batch_test, opts_eval, False)
                total_correct += correct
                total_counts += batch_test[1].size(0)

    net.train()

    if mode == 'openfew':
        closed_samples = opts_eval.n_way*opts_eval.m_query
        open_samples = opts_eval.open_cls*opts_eval.open_sample
        pred_closed_all = torch.cat(pred_closed_list, dim=0)
        target_all = torch.cat(target_list, dim=0)
        open_score_all = torch.cat(open_score_list, dim=0)
        closed_all = torch.cat(closed_list, dim=0)
        accuracy = torch.eq(pred_closed_all, target_all).sum().item() / closed_samples / opts_eval.iterations
        auroc_all = torch.zeros(opts_eval.iterations)
        for i in range(opts_eval.iterations):
            auroc_all[i] = roc_area_calc(dist=open_score_all[(closed_samples+open_samples)*i:(closed_samples+open_samples)*(i+1)],
                                         closed=closed_all[(closed_samples+open_samples)*i:(closed_samples+open_samples)*(i+1)],
                                         descending=True, total_height=open_samples, total_width=closed_samples)
        auroc = auroc_all.mean()
        return accuracy, auroc
    elif mode == 'regular':
        accuracy = total_correct / total_counts
        return accuracy
    else:
        raise NameError('Unknown mode ({})!'.format(mode))


def run_validation(opts, val_db, net, episode, opts_val):

    _curr_str = '\tEvaluating at episode {}, with {} iterations ... (be patient)'.format(episode, opts_val.iterations)
    opts.logger(_curr_str)

    if opts.train.mode.startswith('open'):
        accuracy, auroc = evaluation(net, val_db, opts.train.mode, opts_val)

        eqn = '>' if accuracy > opts.ctrl.best_accuracy else '<'
        _curr_str = '\t\tCurrent accuracy is {:.4f} {:s} ' \
                    'previous best accuracy is {:.4f} (ep{})'.format(accuracy, eqn, opts.ctrl.best_accuracy, opts.ctrl.best_episode)
        opts.logger(_curr_str)
        _curr_str = '\t\tOpen-Set AUROC: {:.4f}'.format(auroc)
        opts.logger(_curr_str)
    elif opts.train.mode == 'regular':
        accuracy = evaluation(net, val_db, opts.train.mode, opts_val)

        eqn = '>' if accuracy > opts.ctrl.best_accuracy else '<'
        _curr_str = '\t\tCurrent accuracy is {:.4f} {:s} ' \
                    'previous best accuracy is {:.4f} (ep{})'.format(accuracy, eqn, opts.ctrl.best_accuracy, opts.val.best_episode)
        opts.logger(_curr_str)
    else:
        raise NameError('Unknown mode ({})!'.format(opts.train.mode))

    if accuracy > opts.ctrl.best_accuracy:
        opts.ctrl.best_accuracy = accuracy
        opts.ctrl.best_episode = episode
        return True
    else:
        return False


def run_test(opts, test_db, net, opts_test):
    _curr_str = 'Evaluating with {} iterations ... (be patient)'.format(opts_test.iterations)
    opts.logger(_curr_str)

    if opts.train.mode.startswith('open'):
        accuracy, auroc = evaluation(net, test_db, opts.train.mode, opts_test)

        _curr_str = '\tAccuracy: {:.4f}'.format(accuracy)
        opts.logger(_curr_str)
        _curr_str = '\tOpen-Set AUROC: {:.4f}'.format(auroc)
        opts.logger(_curr_str)
    elif opts.train.mode == 'regular':
        accuracy = evaluation(net, test_db, opts.train.mode, opts_test)

        _curr_str = '\tAccuracy: {:.4f}'.format(accuracy)
        opts.logger(_curr_str)
