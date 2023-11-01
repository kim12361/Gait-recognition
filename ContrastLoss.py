import torch
'''
def Contrast_Losses(out_1,out_2,batch_size):
    temperature = 80
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

'''
from torch import nn
from info_nce import InfoNCE
        
def Contrast_Losses(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1)
    z2 = nn.functional.normalize(z2)

    # 计算相似度
    sim = (z1 @ z2.T) / temperature

    # 计算 loss
    n = sim.size(0)
    loss = 0
    for i in range(n):
        pos_sim = sim[i][i]
        neg_sim = torch.cat((sim[i][:i], sim[i][i+1:]))
        loss += -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim.exp().sum()))
    
    return loss / n

def infoNCE_loss(z1, z2):
    #loss1 = InfoNCE(temperature=0.1, negative_mode = 'paired')
    loss1 = InfoNCE()
    #z1 = z1.unsqueeze(1)
    #z2 = z2.unsqueeze(1)
   
    loss = loss1(z1,z2)
    return loss   
  
