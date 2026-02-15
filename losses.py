import torch
import torch.nn.functional as F
import torch.nn as nn


def compute_distillation_loss(student_logits, teacher_logits, temp=2.0):
    """
        stu_logits: pred_node_aux, shape (Batch_Size, Num_Classes)
        tea_logits: pred, shape (Batch_Size, Num_Classes)
        temp: Temperature coefficient
    """
    # Detach teacher to stop gradients propagating back to the main head
    teacher_logits = teacher_logits.clone().detach()
    teacher_probs = F.softmax(teacher_logits / temp, dim=1)
    student_log_probs = F.log_softmax(student_logits / temp, dim=1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)

    return kl_loss * (temp * temp)


def compute_cross_entropy_Orthogonal_Regularization_loss(x):
    batch_size, num_tokens, token_dim = x.size()
    # x_norm : (B, N, D)
    x_norm = F.normalize(x, p=2, dim=2)
    # (B, N, D) @ (B, D, N) -> (B, N, N)
    # x_norm.transpose(1, 2) shape (B, D, N)
    similarity_matrices = torch.bmm(x_norm, x_norm.transpose(1, 2))
    # similarity_matrices (B, N, N)

    ground_truth = torch.arange(num_tokens, dtype=torch.long, device=x.device)
    ground_truth = ground_truth.unsqueeze(0).expand(batch_size, num_tokens)

    regularizer_loss = torch.nn.CrossEntropyLoss(reduction='mean')(similarity_matrices,
                                                                   ground_truth)  # B x num_sg

    return regularizer_loss