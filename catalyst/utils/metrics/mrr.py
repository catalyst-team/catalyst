import torch

def mrr(
    outputs: torch.Tensor,
    targets: torch.Tensor
)

    """
    Calculate the MRR score given model ouptputs and targets
    Args:
        outputs [batch_size, slate_length] (torch.Tensor): model outputs, logits
        targets [batch_szie, slate_length] (torch.Tensor): ground truth, labels
    Returns:
        mrr (float): the mrr score
    """
    outputs = outputs.clone()
    targets = targets.clone()
    
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr