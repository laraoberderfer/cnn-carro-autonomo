'''
A primeira etapa é determinar se a GPU deve ser usada ou não. Um padrão comum é usar o argparsemódulo Python para ler os argumentos do usuário e ter um sinalizador que pode ser usado para desabilitar CUDA, em combinação com is_available(). A seguir, args.deviceresulta em um torch.deviceobjeto que pode ser usado para mover tensores para CPU ou CUDA.
'''

import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
