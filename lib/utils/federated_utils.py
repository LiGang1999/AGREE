from itertools import permutations, combinations
import torch
import numpy as np
import torch.nn as nn


def create_domain_weight(source_domain_num):
    global_federated_matrix = [1 / (source_domain_num + 1)] * (source_domain_num + 1)
    return global_federated_matrix


def update_domain_weight(global_domain_weight, epoch_domain_weight, momentum=0.9):
    global_domain_weight = [round(global_domain_weight[i] * momentum + epoch_domain_weight[i] * (1 - momentum), 4)
                            for i in range(len(epoch_domain_weight))]
    return global_domain_weight


def federated_average(model_list, coefficient_matrix, batchnorm_mmd=True):
    """
    :param model_list: a list of all models needed in federated average. [0]: model for target domain,
    [1:-1] model for source domains
    :param coefficient_matrix: the coefficient for each model in federate average, list or 1-d np.array
    :param batchnorm_mmd: bool, if true, we use the batchnorm mmd
    :return model list after federated average
    """
    if batchnorm_mmd:
        dict_list = [it.state_dict() for it in model_list]
        dict_item_list = [dic.items() for dic in dict_list]
        for key_data_pair_list in zip(*dict_item_list):
            source_data_list = [pair[1] * coefficient_matrix[idx] for idx, pair in
                                enumerate(key_data_pair_list)]
            dict_list[0][key_data_pair_list[0][0]] = sum(source_data_list)
        for model in model_list:
            model.load_state_dict(dict_list[0])
    else:
        named_parameter_list = [model.named_parameters() for model in model_list]
        for parameter_list in zip(*named_parameter_list):
            source_parameters = [parameter[1].data.clone() * coefficient_matrix[idx] for idx, parameter in
                                 enumerate(parameter_list)]
            parameter_list[0][1].data = sum(source_parameters)
            for parameter in parameter_list[1:]:
                parameter[1].data = parameter_list[0][1].data.clone()


def adaptively_attack_with_AE(model, classifier, data, target, epoch, total_epochs, prob):
    model.eval()
    classifier.eval()
    tag = [1] * data.size(0)
    # epsilon = (6.0 / 255.0) + (4.0 / 255.0) * (epoch / total_epochs)
    epsilon = (6.0 / 255.0) + (6.0 / 255.0) * (epoch / total_epochs)
    epsilon_list = [epsilon] * data.size(0)
    for i in range(data.size(0)):
        epsilon_list[i] = epsilon_list[i] * (1.0 - prob[i].item())
    num_steps = 2
    step_size_list = [0] * len(epsilon_list)
    for i in range(data.size(0)):
        step_size_list[i] = epsilon_list[i] / 10
    x_adv = data.detach()
    for i in range(data.size(0)):
        x_adv[i] = x_adv[i] + torch.from_numpy(np.random.uniform(-epsilon_list[i], epsilon_list[i], data[i].shape)).float().cuda()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = classifier(model(x_adv))
        _, predict = output.max(1)
        for i in range(data.size(0)):
            if tag[i] > 0.5 and predict[i] != target[i]:
                tag[i] = 0
        if k == num_steps - 1:
            break
        with torch.enable_grad():
            loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
        loss_adv.backward()
        eta = x_adv.grad.sign()
        for i in range(data.size(0)):
            eta[i] = step_size_list[i] * eta[i]
        x_adv = x_adv.detach() + eta
        for i in range(data.size(0)):
            x_adv[i] = torch.min(torch.max(x_adv[i], data[i] - epsilon_list[i]), data[i] + epsilon_list[i])
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    classifier.train()
    return tag


def fastly_PL_denoise_with_AE(raw_knowledge_list, knowledge_list, num_classes, image_t, model_list, classifier_list, epoch, total_epochs, confidence_gate):
    konwledge_contribution = [0] * knowledge_list.size(1)
    max_p, max_p_class = knowledge_list.max(2)
    max_p_mask = torch.ones(knowledge_list.size(0), knowledge_list.size(1)).cuda()
    abandon, total = 0, knowledge_list.size(0) * knowledge_list.size(1)
    for j in range(knowledge_list.size(1)):
        each_model_max_p, each_model_max_p_class = raw_knowledge_list[j].max(1)
        use_confidence_gate = False
        if use_confidence_gate:# Confidence threshold is an efficient strategy.
            tag = [0] * image_t.size(0)
            for i in range(image_t.size(0)):
                if max_p[i][j] > confidence_gate:
                    tag[i] = 1
        else:# Adversarial example is an alternative solution.
            tag = adaptively_attack_with_AE(model_list[j+1], classifier_list[j+1], image_t,
                                            each_model_max_p_class, epoch, total_epochs, each_model_max_p)
        for i in range(knowledge_list.size(0)):
            if tag[i] < 0.5:
                max_p_mask[i][j] = 0.0
                abandon += 1
    consensus_knowledge_mask = torch.zeros(knowledge_list.size(0)).cuda()
    _, consensus_knowledge = max_p.max(1)
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        if torch.sum(p_mask) < 0.5:###
            cnt = torch.zeros(num_classes).cuda()
            for source_idx, source_class in enumerate(p_class):
                cnt[source_class] += p[source_idx]
            _, consensus_knowledge[batch_idx] = cnt.max(0)
        else:
            cnt = torch.zeros(num_classes).cuda()
            for source_idx, source_class in enumerate(p_class):
                if p_mask[source_idx] < 0.5:
                    continue
                cnt[source_class] += p[source_idx]
            _, consensus_knowledge[batch_idx] = cnt.max(0)
            consensus_knowledge_mask[batch_idx] = 1.0
            for source_idx, source_class in enumerate(p_class):
                if source_class == consensus_knowledge[batch_idx] and p_mask[source_idx] > 0.5:
                    konwledge_contribution[source_idx] += 1
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).cuda().scatter_(1,
                                                                                                consensus_knowledge.view(
                                                                                                    -1, 1), 1)
    return konwledge_contribution, consensus_knowledge, consensus_knowledge_mask, abandon, total


def knowledge_vote(knowledge_list, confidence_gate, num_classes):
    """
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :return: consensus_confidence,consensus_knowledge,consensus_knowledge_weight
    """
    max_p, max_p_class = knowledge_list.max(2)
    max_conf, _ = max_p.max(1)
    max_p_mask = (max_p > confidence_gate).float().cuda()
    consensus_knowledge = torch.zeros(knowledge_list.size(0), knowledge_list.size(2)).cuda()
    for batch_idx, (p, p_class, p_mask) in enumerate(zip(max_p, max_p_class, max_p_mask)):
        # to solve the [0,0,0] situation
        if torch.sum(p_mask) > 0:
            p = p * p_mask
        for source_idx, source_class in enumerate(p_class):
            consensus_knowledge[batch_idx, source_class] += p[source_idx]
    consensus_knowledge_conf, consensus_knowledge = consensus_knowledge.max(1)
    consensus_knowledge_mask = (max_conf > confidence_gate).float().cuda()
    consensus_knowledge = torch.zeros(consensus_knowledge.size(0), num_classes).cuda().scatter_(1,
                                                                                                consensus_knowledge.view(
                                                                                                    -1, 1), 1)
    return consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask


def calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate, source_domain_numbers,
                              num_classes):
    """
    :param consensus_focus_dict: record consensus_focus for each domain
    :param torch.tensor knowledge_list : recording the knowledge from each source domain model
    :param float confidence_gate: the confidence gate to judge which sample to use
    :param source_domain_numbers: the numbers of source domains
    """
    domain_contribution = {frozenset(): 0}
    for combination_num in range(1, source_domain_numbers + 1):
        combination_list = list(combinations(range(source_domain_numbers), combination_num))
        for combination in combination_list:
            consensus_knowledge_conf, consensus_knowledge, consensus_knowledge_mask = knowledge_vote(
                knowledge_list[:, combination, :], confidence_gate, num_classes)
            domain_contribution[frozenset(combination)] = torch.sum(
                consensus_knowledge_conf * consensus_knowledge_mask).item()
    permutation_list = list(permutations(range(source_domain_numbers), source_domain_numbers))
    permutation_num = len(permutation_list)
    for permutation in permutation_list:
        permutation = list(permutation)
        for source_idx in range(source_domain_numbers):
            consensus_focus_dict[source_idx + 1] += (
                                                            domain_contribution[frozenset(
                                                                permutation[:permutation.index(source_idx) + 1])]
                                                            - domain_contribution[
                                                                frozenset(permutation[:permutation.index(source_idx)])]
                                                    ) / permutation_num
    return consensus_focus_dict


def decentralized_training_strategy(communication_rounds, epoch_samples, batch_size, total_epochs):
    """
    Split one epoch into r rounds and perform model aggregation
    :param communication_rounds: the communication rounds in training process
    :param epoch_samples: the samples for each epoch
    :param batch_size: the batch_size for each epoch
    :param total_epochs: the total epochs for training
    :return: batch_per_epoch, total_epochs with communication rounds r
    """
    if communication_rounds >= 1:
        epoch_samples = round(epoch_samples / communication_rounds)
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    elif communication_rounds in [0.2, 0.5]:
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    else:
        raise NotImplementedError(
            "The communication round {} illegal, should be 0.2 or 0.5".format(communication_rounds))
    return batch_per_epoch, total_epochs
