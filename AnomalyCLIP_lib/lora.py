import torch
import torch.nn as nn
import torch.nn.functional as F
import math  

class LoRALayer(torch.nn.Module): 
    def __init__(self, in_dim, out_dim, rank, alpha): 
        super().__init__() 
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank,device=torch.device('cuda:0'))) 
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) 
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim,device=torch.device('cuda:0'))) 
        self.alpha = alpha  
    def forward(self, x): 
        x = self.alpha * (x @ self.A @ self.B) 
        return x
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        """返回原始线性层的 weight 属性以兼容 dtype 查询"""
        return self.linear.weight

    @property
    def bias(self):
        """返回原始线性层的 bias 属性（如果存在）"""
        return self.linear.bias
def replace_linear_with_lora(model, rank, alpha): 
    for name, module in model.named_children(): 
        if isinstance(module, torch.nn.Linear): 
            setattr(model, name, LinearWithLoRA(module, rank, alpha)) 
        else: 
            replace_linear_with_lora(module, rank, alpha)

def load_lora_model(model,lora_state):
    lora_state_dict = lora_state['lora_state_dict']
    lora_config = lora_state['lora_config']
    replace_linear_with_lora(model.visual, lora_config['rank'], lora_config['alpha'])
    state_dict = model.state_dict()
    state_dict.update(lora_state_dict)
    model.load_state_dict(state_dict, strict=False)
    return model