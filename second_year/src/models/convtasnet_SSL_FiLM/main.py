# from .CRN import main_model
from .convtasnet import main_model

def get_model(args):

    return main_model(args)
