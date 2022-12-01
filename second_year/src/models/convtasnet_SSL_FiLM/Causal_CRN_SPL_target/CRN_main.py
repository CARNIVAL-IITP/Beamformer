# from .CRN import main_model
from .CRN_SPL_target import main_model
import torch

def get_model(args):

    model=main_model(args)
    trained=torch.load(args['pretrain'])
    model.load_state_dict(trained['model_state_dict'], )
    if args['freeze']:
        for param in model.parameters():
            param.requires_grad=False
   
    return model
