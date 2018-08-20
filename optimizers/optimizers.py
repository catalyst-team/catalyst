import torch
import common.optimizers.shampoo as shampoo
import common.optimizers.openai_adam as openai_adam


OPTIMIZERS = {
    **torch.optim.__dict__,
    **shampoo.__dict__,
    **{"OpenAIAdam": openai_adam.OpenAIAdam}
}
