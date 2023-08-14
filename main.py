import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True, choices=("describe", "evaluate"),
                    help='Specify either describe or evaluate', type=str)
parser.add_argument('--datasets', choices=("sfu", "gardenspointwalking", "stlucia_small"),
                    help='specify one of the datasets from PlaceRec.Datasets', type=str, default="StLucia", nargs='+')
parser.add_argument('--methods', choices=("netvlad", "hog", "cosplace", "hdc_delf", "alexnet"),
                    help="specify one of the techniques from vpr/vpr_tecniques", type=str, default="hog", nargs='+')
parser.add_argument('--batchsize', type=int, default=10, help="Choose the Batchsize for VPR processing")
parser.add_argument('--partition', type=str, default='train', help="choose from 'train', 'val', 'test' or 'all'")
args = parser.parse_args()



def get_method(name: str=None):
        if name == "netvlad":
            from PlaceRec.Methods import NetVLAD
            methods = NetVLAD()
        elif name == "hog":
            from PlaceRec.Methods import HOG
            methods = HOG()
        elif name == "cosplace":
            from PlaceRec.Methods import CosPlace
            methods = CosPlace()
        elif name == "alexnet":
            from PlaceRec.Methods import AlexNet
            methods = AlexNet()
        elif name == "hdc_delf":
            from PlaceRec.Methods import HDC_DELF
            methods = HDC_DELF()
        else:
            raise Exception("Method '" + name + "' not implemented")
    return method


def get_dataset(name: str=None):
    if name == "gardenspointwalking":
        from PlaceRec.Datasets import GardensPointWalking
        dataset = GardensPointWalking()
    if name == "stlucia_small":
        from PlaceRec.Datasets import StLucia_small
        dataset = StLucia_small()
    if name == "sfu":
        from PlaceRec.Datasets import SFU
        dataset = SFU()
    else:
        raise Exception("Dataset '" + name + "' not implemented")
    return dataset

        



if args.mode == "describe":
    for method_name in args.methods:
        for dataset_name in args.datasets:
            method = get_method(method_name)
            ds = get_dataset(dataset_name)

            """ To Do: I do not like the return of a partial dictionary from the 
                method class. Please change it so the method class accepts a dataloader """

elif args.mode == "eval":
    """ perform the evaluation and save it """