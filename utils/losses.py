import torch.nn as nn
import torch.nn.functional as F


def clip_kl(student_dict, teacher_dict):
    """KL divergence loss.
    """
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # input should be a distribution in the log space
    student_logits = F.log_softmax(student_dict['clipwise_output'])
    teacher_logits = F.softmax(teacher_dict['clipwise_output'])
    return kl_loss(student_logits, teacher_logits)


def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def clip_ce(output_dict, target_dict):
    """Crossentropy loss.
    """
    return F.cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'clip_ce':
        return clip_ce
    elif loss_type == 'clip_kl':
        return clip_kl
