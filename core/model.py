import torch
from torch import nn
from torch.nn import functional as F

from core.feat import feat_extract, BasicBlock, conv1x1


class OpenNet(nn.Module):
    def __init__(self, opts):
        super(OpenNet, self).__init__()

        self.opts = opts
        self.num_classes = opts.model.num_classes

        self._norm_layer = nn.BatchNorm2d

        opts.logger('Building up models ...')
        # feature extractor
        if opts.train.mode == 'regular':
            self.feat_net, self.block_expansion = feat_extract(structure=opts.model.structure, branch=False)
        elif opts.train.open_detect == 'center':
            self.feat_net, self.block_expansion = feat_extract(structure=opts.model.structure, branch=False)
        else:
            self.feat_net, self.block_expansion = feat_extract(structure=opts.model.structure, branch=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sm = nn.Softmax(dim=1)

        self.fc = nn.Linear(512 * self.block_expansion, self.num_classes)

        if opts.train.mode == 'openfew':
            self.cel_all = nn.CrossEntropyLoss()
            block = BasicBlock
            self.inplanes = 512 * self.block_expansion
            self.layer_sigs_0 = self._make_layer(block, 512, 2, stride=1)
            self.inplanes = 512 * block.expansion * 2
            self.layer_sigs_1 = self._make_layer(block, 512, 2, stride=1)

            if self.opts.train.aux:
                self.fc = nn.Linear(512 * self.block_expansion, self.num_classes)

        elif opts.train.mode == 'openmany':
            self.cel_all = nn.CrossEntropyLoss()
            block = BasicBlock
            self.inplanes = 512 * self.block_expansion
            self.layer_sigs_0 = self._make_layer(block, 512, 2, stride=1)
            self.inplanes = 512 * block.expansion * 2
            self.layer_sigs_1 = self._make_layer(block, 512, 2, stride=1)

        elif opts.train.mode == 'regular':
            self.cel_all = nn.CrossEntropyLoss()
            self.fc = nn.Linear(512 * self.block_expansion, self.num_classes)
        else:
            raise NameError('Unknown mode ({})!'.format(opts.train.mode))

    def forward(self, batch, opts_runtime, train=True):

        if self.opts.train.mode == 'openmany':
            return self.forward_openmany(batch, opts_runtime, train)
        if self.opts.train.mode == 'openfew':
            return self.forward_openfew(batch, opts_runtime, train)
        if self.opts.train.mode == 'regular':
            return self.forward_regular(batch, train)

    def forward_openfew(self, batch, opts_runtime, train=True):
        input, target = batch[0].to(self.opts.ctrl.device), batch[1].to(self.opts.ctrl.device)
        if len(input.size()) > 4:
            c = input.size(-3)
            h = input.size(-2)
            w = input.size(-1)
            input = input.view(-1, c, h, w)

        n_way = opts_runtime.n_way
        k_shot = opts_runtime.k_shot
        m_query = opts_runtime.m_query
        open_cls = opts_runtime.open_cls
        open_sample = opts_runtime.open_sample
        aug_scale = opts_runtime.aug_scale
        fold = opts_runtime.fold

        support_amount = n_way * k_shot
        query_amount = int(n_way * m_query / fold)
        open_amount = int(open_cls * open_sample / fold)

        # FEATURE EXTRACTION
        x_all = self.feat_net(input)
        if self.opts.train.open_detect == 'center':
            x_mu = x_all
        else:
            x_mu = x_all[0]
            x_sigs = x_all[1]

        # LOSS OR ACCURACY
        if train:
            # prepare class gauss
            support_mu = x_mu[:support_amount, :, :, :]
            query_mu = x_mu[support_amount:support_amount+query_amount+open_amount, :, :, :]
            if self.opts.train.open_detect == 'gauss':
                support_sigs_0 = x_sigs[:support_amount, :, :, :]
                support_sigs_1 = self.layer_sigs_0(support_sigs_0).mean(dim=0, keepdim=True).expand_as(support_sigs_0)
                support_sigs_1 = torch.cat((support_sigs_0, support_sigs_1), dim=1)
                support_sigs = self.layer_sigs_1(support_sigs_1)

            target_unique, target_fsl = self.get_fsl_target(target[:support_amount+query_amount])
            target_support = target_fsl[:support_amount]
            target_query = target_fsl[support_amount:support_amount+query_amount]

            batch_size, feat_size, feat_h, feat_w = query_mu.size()

            # fewshot loss
            idxs = torch.stack([(target_support == i).nonzero().squeeze() for i in range(self.opts.fsl.n_way)])
            if len(idxs.size()) < 2:
                idxs.unsqueeze_(1)
            mu = support_mu[idxs, :, :, :].mean(dim=1)
            if self.opts.train.open_detect == 'center':
                mu_whitten = mu
                query_mu_whitten = query_mu
            else:
                sigs = support_sigs[idxs, :, :, :].mean(dim=1)
                mu_whitten = torch.mul(mu, sigs)
                query_mu_whitten = torch.mul(query_mu.unsqueeze(1), sigs.unsqueeze(0))
            mu_whitten = self.avgpool(mu_whitten)
            mu_whitten = mu_whitten.view(-1, feat_size)
            query_mu_whitten = self.avgpool(query_mu_whitten.view(-1, feat_size, feat_h, feat_w))
            query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
            dist_few = -torch.norm(query_mu_whitten - mu_whitten.unsqueeze(0), p=2, dim=2)
            dist_few_few = dist_few[:query_amount, :]

            l_few = self.cel_all(dist_few_few, target_query)

            # openfew loss
            if self.opts.train.entropy:
                dist_few_open = dist_few[query_amount:query_amount+open_amount, :]

                loss_open = F.softmax(dist_few_open, dim=1) * F.log_softmax(dist_few_open, dim=1)
                loss_open = loss_open.sum(dim=1)
                l_open = loss_open.mean()
            else:
                l_open = torch.tensor([0])

            if self.opts.train.aux:
                target_base = target[support_amount+query_amount+open_amount:]
                base_mu = x_mu[support_amount+query_amount+open_amount:, :, :, :]
                dist_base = self.avgpool(base_mu)
                dist_base = dist_base.view(-1, feat_size)
                cls_pred = self.fc(dist_base)
                l_aux = self.cel_all(cls_pred, target_base)
            else:
                l_aux = torch.tensor([0])

            if self.opts.train.entropy and self.opts.train.aux:
                loss = l_few + l_open * self.opts.train.loss_scale_entropy + l_aux * self.opts.train.loss_scale_aux
            elif self.opts.train.entropy:
                loss = l_few + l_open * self.opts.train.loss_scale_entropy
            elif self.opts.train.aux:
                loss = l_few + l_aux * self.opts.train.loss_scale_aux
            else:
                loss = l_few

            return loss

        else:
            # TEST
            # prepare class gauss
            support_mu = x_mu[:support_amount*aug_scale, :, :, :]
            query_mu = x_mu[support_amount*aug_scale:(support_amount+query_amount+open_amount)*aug_scale, :, :, :]
            if self.opts.train.open_detect == 'gauss':
                support_sigs_0 = x_sigs[:support_amount*aug_scale, :, :, :]
                support_sigs_1 = self.layer_sigs_0(support_sigs_0).mean(dim=0, keepdim=True).expand_as(support_sigs_0)
                support_sigs_1 = torch.cat((support_sigs_0, support_sigs_1), dim=1)
                support_sigs = self.layer_sigs_1(support_sigs_1)

            target_unique, target_fsl = self.get_fsl_target(target[:support_amount+query_amount])
            target_support = target_fsl[:support_amount]
            target_query = target_fsl[support_amount:support_amount+query_amount]

            batch_size, feat_size, feat_h, feat_w = query_mu.size()

            # fewshot
            support_mu = support_mu.view(-1, aug_scale, feat_size, feat_h, feat_w).mean(dim=1)
            if self.opts.train.open_detect == 'gauss':
                support_sigs = support_sigs.view(-1, aug_scale, feat_size, feat_h, feat_w).mean(dim=1)
            idxs = torch.stack([(target_support == i).nonzero().squeeze() for i in range(n_way)])
            if len(idxs.size()) < 2:
                idxs.unsqueeze_(1)
            mu = support_mu[idxs, :, :, :].mean(dim=1)
            if self.opts.train.open_detect == 'center':
                mu_whitten = mu
                query_mu_whitten = query_mu
            else:
                sigs = support_sigs[idxs, :, :, :].mean(dim=1)
                mu_whitten = torch.mul(mu, sigs)
                query_mu_whitten = torch.mul(query_mu.unsqueeze(1), sigs.unsqueeze(0))
            mu_whitten = self.avgpool(mu_whitten)
            mu_whitten = mu_whitten.view(-1, feat_size)
            query_mu_whitten = self.avgpool(query_mu_whitten.view(-1, feat_size, feat_h, feat_w))
            query_mu_whitten = query_mu_whitten.view(batch_size, -1, feat_size)
            dist_few = torch.norm(query_mu_whitten - mu_whitten.unsqueeze(0), p=2, dim=2)

            dist_few = dist_few.view(-1, aug_scale, n_way).mean(dim=1)
            dist_few_sm = self.sm(dist_few)
            all_score, all_pred = dist_few_sm.min(dim=1)
            few_pred = all_pred[:query_amount]
            closed_few = torch.ones(query_amount)
            closed_open = -torch.ones(open_amount)
            closed = torch.cat((closed_few, closed_open), dim=0)
            return few_pred.detach().cpu(), target_query.detach().cpu(), all_score.detach().cpu(), closed

    def forward_regular(self, batch, train=True):
        input, target = batch[0].to(self.opts.ctrl.device), batch[1].to(self.opts.ctrl.device)

        # FEATURE EXTRACTION
        x = self.feat_net(input)
        x = self.avgpool(x)
        N = x.size(0)
        x = x.view(N, -1)

        # LOSS OR ACCURACY
        if train:
            cls_pred = self.fc(x)
            loss = self.cel_all(cls_pred, target)
            return loss
        else:
            # TEST
            cls_pred = self.fc(x)
            _, pred_cls = cls_pred.max(dim=1)
            correct = torch.eq(pred_cls, target)
            return correct.sum().item()

    def get_fsl_target(self, target):
        target_unique, target_fsl = target.unique(return_inverse=True)
        return target_unique, target_fsl

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
