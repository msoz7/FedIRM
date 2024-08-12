import torch
import numpy as np
import torch.nn.functional as F
import copy
from options import args_parser

args = args_parser()

NUM_CLASSES = args.num_classes

temperature = 2.0


def torch_tile(tensor, dim, n):
    if dim == 0:
        return (
            tensor.unsqueeze(0)
            .transpose(0, 1)
            .repeat(1, n, 1)
            .view(-1, tensor.shape[1])
        )
    else:
        return (
            tensor.unsqueeze(0)
            .transpose(0, 1)
            .repeat(1, 1, n)
            .view(tensor.shape[0], -1)
        )



def jensen_shannon_divergence(p, q,eps=1e-12):
    """
    Calculate Jensen-Shannon Divergence between two probability distributions.

    Parameters:
    p (numpy.ndarray): First probability distribution.
    q (numpy.ndarray): Second probability distribution.

    Returns:
    float: Jensen-Shannon Divergence.
    """
    # with torch.no_grad():
    #     p=p.cpu()
    #     q=q.cpu()
    #     m = (p + q) / 2
    #     kl_p = torch.sum(p * np.log2(p / m))
    #     kl_q = torch.sum(q * np.log2(q / m))
    #     jsd = (kl_p + kl_q) / 2
    p=F.softmax(p, dim=0)
    m = (p + q) / 2
    
    # Compute the Jensen-Shannon Divergence
    jsd = 0.5 * (torch.sum(p * torch.log2(p / m)) + torch.sum(q * torch.log2(q / m)))
    #jsd = torch.sum(p * torch.log(p / q))
    return jsd


def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.

    Parameters:
    vector1 (numpy.ndarray): First vector.
    vector2 (numpy.ndarray): Second vector.

    Returns:
    float: Cosine similarity between the two vectors.
    """
    with torch.no_grad():
        dot_product = np.dot(vector1.cpu(), vector2.cpu())
        norm_vector1 = np.linalg.norm(vector1.cpu())
        norm_vector2 = np.linalg.norm(vector2.cpu())
    
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0  # Prevent division by zero
    
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    
    return cosine_similarity
def cosine_similarity_matrix(outputs, averaged_class_matrix):
    """
    Calculate cosine similarity between each output vector and averaged class matrix.

    Parameters:
    outputs (numpy.ndarray): Output probability vectors for each sample.
    averaged_class_matrix (numpy.ndarray): Averaged class matrix.

    Returns:
    numpy.ndarray: Cosine similarity matrix.
    """
    num_samples = outputs.shape[0]
    num_classes = averaged_class_matrix.shape[0]
    cosine_sim_matrix = np.zeros((num_samples, num_classes))
    
    for i in range(num_samples):
        for j in range(num_classes):
            #cosine_sim_matrix[i, j] = cosine_similarity(outputs[i], averaged_class_matrix[j])
            cosine_sim_matrix[i, j] = jensen_shannon_divergence(outputs[i], averaged_class_matrix[j])
            # with torch.no_grad():
            #     cosine_sim_matrix[i, j] = np.corrcoef(outputs[i].cpu(), averaged_class_matrix[j].cpu())[0, 1]
    
    return cosine_sim_matrix

def update_pseudo_labels(outputs, averaged_class_matrix, old_pseudo_labels):
    """
    Update pseudo-labels based on cosine similarity.

    Parameters:
    outputs (numpy.ndarray): Output probability vectors for each sample.
    averaged_class_matrix (numpy.ndarray): Averaged class matrix.
    old_pseudo_labels (numpy.ndarray): Old pseudo-labels.

    Returns:
    numpy.ndarray: Updated pseudo-labels.
    """
    cosine_sim_matrix = cosine_similarity_matrix(outputs, averaged_class_matrix)
    new_pseudo_labels = copy.deepcopy(old_pseudo_labels)
    counter=0
    for i in range(len(outputs)):
        
        max_sim_idx = np.argmin(cosine_sim_matrix[i])
        max_sim_value = cosine_sim_matrix[i, max_sim_idx]
        #print(max_sim_value)
        
        if max_sim_value<cosine_sim_matrix[i, old_pseudo_labels[i]]:
        #if cosine_sim_matrix[i, old_pseudo_labels[i]] > 0.22 and max_sim_value<cosine_sim_matrix[i, old_pseudo_labels[i]]:

            counter=counter+1
            new_pseudo_labels[i] = max_sim_idx
    
    return new_pseudo_labels,counter
    
    
def get_confuse_matrix(logits, labels):
    source_prob = []

    for i in range(NUM_CLASSES):
        mask = torch_tile(torch.unsqueeze(labels[:, i], -1), 1, args.mean_used_features)
        logits_mask_out = logits * mask
        logits_avg = torch.sum(logits_mask_out, dim=0) / (
            torch.sum(labels[:, i]) + 1e-8
        )
        prob = F.softmax(logits_avg / temperature, dim=0)
        source_prob.append(prob)
    
    return torch.stack(source_prob)


def get_confuse_covariance(logits, labels):

    cov_matrix = []


    for i in range(NUM_CLASSES):

        mask = torch_tile(torch.unsqueeze(labels[:, i], -1), 1, NUM_CLASSES)
        logits_mask_out = logits * mask
        logits_avg = torch.sum(logits_mask_out, dim=0) / (
            torch.sum(labels[:, i]) + 1e-8
        )

        difference = logits_mask_out - logits_avg


        covariance_matrix  = torch.mm(difference.t(), difference)

        covariance_matrix /= (torch.sum(labels[:, i]) + 1e-8)

        cov_mean = torch.mean(covariance_matrix)
        cov_std = (torch.std(covariance_matrix) + 1e-8)
        
        covariance_matrix = (covariance_matrix - cov_mean) / cov_std


        max_value = torch.quantile(covariance_matrix, 0.90)
        min_value = torch.quantile(covariance_matrix, 0.25)
        #
        covariance_matrix = torch.clamp(covariance_matrix, min=min_value, max=max_value)


        covariance_matrix = covariance_matrix.flatten()
        # print(f"covsize-postvec: {covariance_matrix.size()}")

        covariance_matrix = F.softmax(covariance_matrix / temperature, dim=0)


        cov_matrix.append(covariance_matrix)

        # covariance_matrix = torch.mean(covariance_matrix, dim=0)

    #stack = torch.stack(cov_matrix)
    #print(f"covsize-afterstack: {stack.size()}")
    return torch.stack(cov_matrix)

def kd_loss(source_matrix, target_matrix):
    loss_fn = torch.nn.MSELoss(reduction="none")
    eps=1e-12
    Q = source_matrix+eps
    P = target_matrix+eps
    #Q = source_matrix.t()+eps
    #P = target_matrix.t()+eps
    
    loss = (
        F.kl_div(P.log(), Q, None, None, "batchmean")+
        F.kl_div(Q.log(), P, None, None, "batchmean")
       
    )/2
    #print(f"kl loss{loss}")
    return  torch.sum(loss)

