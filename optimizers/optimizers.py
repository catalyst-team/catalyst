import torch
import prometheus.optimizers.shampoo as shampoo
import prometheus.optimizers.openai_adam as openai_adam


OPTIMIZERS = {
    **torch.optim.__dict__,
    **shampoo.__dict__,
    **{"OpenAIAdam": openai_adam.OpenAIAdam}
}
