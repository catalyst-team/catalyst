import torch
import catalyst.legacy.optimizers.shampoo as shampoo
import catalyst.legacy.optimizers.openai_adam as openai_adam

OPTIMIZERS = {
    **torch.optim.__dict__,
    **shampoo.__dict__,
    **{
        "OpenAIAdam": openai_adam.OpenAIAdam
    }
}
