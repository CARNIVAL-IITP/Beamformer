# from .CRN import main_model
# from .EABNET import EaBNet
# from .total_EABNET import Total_model
from .fspen_total import Total_model

def get_model(args):


    return Total_model(args)
