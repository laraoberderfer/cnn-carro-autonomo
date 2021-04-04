from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision import transforms
import os

def carrega_imagem(path):
    with open(path, 'rb') as frame:
        imagem = Image.open(frame)
        return imagem.convert('RGB')

# transforma imagem para tensor
transforma_imagem = transforms.Compose([
    transforms.CenterCrop(72),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# classe para tratamento das imagens
class LoteImagens:
    def __init__(self, dataset_path):
        # abre imagens da pasta
        self.images = os.path.join(dataset_path, "imagens")

        # abre o arquivo rotulos para leitura
        with open(os.path.join(dataset_path, "rotulos.txt"), 'r') as f:
            lines = [l.strip().split() for l in f.readlines()]
            lines = [[f, int(label)] for (f, label) in lines]
            self.labels = lines

        # carrega imagem pronta para o tensor
        self.transform = transforma_imagem
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name, label = self.labels[index]
        return self.transform(carrega_imagem(os.path.join(self.images, image_name))), torch.LongTensor([label])


def get_lote(pasta, tamanho):
    # recebendo lote de imagens
    lote = LoteImagens(pasta)

    # embaralhando lote de imagens
    return DataLoader(lote, batch_size=tamanho, shuffle=True)

# testando a saida da imagem
def testandoImagens():
    lote = LoteImagens("imagens")
    print("Total de imagens: ", len(lote))
    print(lote[0])

LEFT = 0    # esquerda
RIGHT = 1   # direita
GO = 2      # aceleração

ACTIONS = [LEFT, RIGHT, GO]

if __name__=='__main__':
    testandoImagens()