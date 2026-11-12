import torch 

def compute_uncer(pred_out):

    uncer_out = torch.softmax(pred_out, dim=1)
    ## 计算学习比重
    uncer_out = torch.sum(-uncer_out * torch.log(uncer_out), dim=1, keepdim=True)

    return uncer_out




