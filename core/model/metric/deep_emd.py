import torch
import torch.nn as nn
from .metric_model import MetricModel
from core.model.backbone.utils.deep_emd import emd_inference_opencv, emd_inference_qpth
from core.model.backbone.resnet_12_emd import ResNet
import torch.nn.functional as F
from core.utils import accuracy


class Network(MetricModel):

    def __init__(self, mode, args, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.mode = mode
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.encoder = ResNet()
        self.k = self.args.get("way") * self.args.get("shot")

        if self.mode == 'pre_train':
            self.fc = nn.Linear(640, self.args.num_class)

    def reshuffle_data(self, data_shot, labels, cls):
        label_len = labels.__len__()
        per_len = label_len // cls
        idxs = torch.arange(label_len) // cls + (torch.arange(label_len) % cls) * per_len
        return data_shot[idxs], labels[idxs]

    def forward_output(self, logits):
        return torch.argmax(logits, dim=1)

    def forward(self, batch):
        if self.training:
            return self.set_forward_loss(batch)
        else:
            return self.set_forward(batch)

    def set_forward(self, batch):
        image, _ = batch
        image = image.to(self.device)
        data = self.encode(image)
        data_shot, data_query = data[:self.k], data[self.k:]
        label = torch.arange(self.args.get("way")).repeat(self.args.get("query"))
        label = label.type(torch.cuda.LongTensor)
        if self.args.get("shot") > 1:
            data_shot = self.get_sfc(data_shot)
        logits = self.set_forward_adaptation(data_shot.unsqueeze(0).repeat(1, 1, 1, 1, 1), data_query)

        acc = accuracy(logits, label)
        output = self.forward_output(logits)
        return output, acc

    def set_forward_loss(self, batch):
        image, global_target = batch
        image = image.to(self.device)
        global_target = global_target.to(self.device)

        if self.mode == 'pre_train':
            image, global_target = self.reshuffle_data(image, global_target.reshape(-1), self.way_num)
            feat = self.emb_func(image)
            label = global_target

            logits = self.fc(feat.squeeze(-1).squeeze(-1))

        else:
            data = self.encode(image)
            data_shot, data_query = data[:self.k], data[self.k:]
            label = torch.arange(self.args.get("way")).repeat(self.args.get("query"))
            label = label.type(torch.cuda.LongTensor)

            if self.args.get("shot") > 1:
                data_shot = self.get_sfc(data_shot)
            logits = self.set_forward_adaptation(data_shot.unsqueeze(0).repeat(1, 1, 1, 1, 1), data_query)
            output = self.forward_output(logits)

        loss = self.loss_func(logits, label)
        acc = accuracy(logits, label)
        return output, acc, loss

    def set_forward_adaptation(self, proto, query):
        proto = proto.squeeze(0)
        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.get("solver") == 'opencv' or (not self.training):
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def pre_train_forward(self, _input):
        return self.fc(self.encode(_input, dense=False).squeeze(-1).squeeze(-1))

    def get_weight_vector(self, A, B):
        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the proto
        SFC = support.view(self.args.get("shot"), -1, 640, support.shape[-2], support.shape[-1]).mean(
            dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.args.get("sfc_lr"), momentum=0.9, dampening=0.9, weight_decay=0)

        label_shot = torch.arange(self.args.get("way")).repeat(self.args.get("shot"))
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.get("sfc_update_step")):
                rand_id = torch.randperm(self.args.get("way") * self.args.get("shot")).cuda()
                for j in range(0, self.args.get("way") * self.args.get("shot"), self.args.get("sfc_bs")):
                    selected_id = rand_id[
                                  j: min(j + self.args.get("sfc_bs"), self.args.get("way") * self.args.get("shot"))]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):

        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        _num_node = weight_1.shape[-1]
        if solver == 'opencv':  # use openCV solver
            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
                    similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

            temperature = (self.args.get("temperature") / _num_node)
            logitis = similarity_map.sum(-1).sum(-1) * temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2, form=self.args.get("form"),
                                          l2_strength=self.args.get("l2_strength"))

            logitis = (flows * similarity_map).view(num_query, num_proto, flows.shape[-2], flows.shape[-1])
            temperature = (self.args.get("temperature") / _num_node)
            logitis = logitis.sum(-1).sum(-1) * temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.get("norm") == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similiarity_map(self, proto, query):

        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args.get("metric") == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)

        if self.args.get("metric") == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

    def encode(self, x, dense=True):

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x

        else:
            x = self.encoder(x)
            if not dense:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.args.get("feature_pyramid") is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.args.get("feature_pyramid"):
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out


class DeepEMD(MetricModel):
    def __init__(self, mode, args, **kwargs):
        super(DeepEMD, self).__init__(**kwargs)
        self.deep_emd= Network(mode, args, **kwargs)
        print("trying to read model----------")
        print("pretrain_path",args.get("pretrain_path"))
        if args.get("pretrain_path") is not None:
            print("----------------------model loaded----------------------------------------------------")
            self.deep_emd.load_state_dict(torch.load(args.get("pretrain_path")))
    
    def forward(self, batch):
        return self.deep_emd.forward(batch)
    def set_forward_loss(self, batch):
        return self.deep_emd.set_forward_loss(batch)
    
    def set_forward(self, batch):
        return self.deep_emd.set_forward(batch)
    
    
    