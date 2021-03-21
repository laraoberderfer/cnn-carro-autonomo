#https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
#https://gist.github.com/PulkitS01/4b467252bbe0fbf520d91f3608e2284d#file-library-py
#https://nestedsoftware.com/2019/09/09/pytorch-image-recognition-with-convolutional-networks-4k17.159805.html
# https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-translate-sign-language-into-english-pt
# https://www.fulljoin.com.br/posts/2020-04-27-usando-o-pytorch-no-r-treinando-o-seefood/
# https://nestedsoftware.com/2019/09/09/pytorch-image-recognition-with-convolutional-networks-4k17.159805.html


# importing the libraries
import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
#matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
#from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

'''
Função para treinar o modelo
''' 
def treinando(epoch):
    model.train()
    tr_loss = 0.0

    # obtendo o conjunto de treinamento
    x_train, y_train = Variable(train_x), Variable(train_y)   
    # obtendo o conjunto de validação
    x_val, y_val = Variable(val_x), Variable(val_y)    

    # converter os dados em formato GPU
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # limpar os gradientes dos parâmetros do modelo
    optimizer.zero_grad()
    
    print('antes de treinar:')
    print(train_x.shape)
    print(val_x.shape)
    print(train_y.shape)
    print(val_y.shape)

    # previsão para treinamento e conjunto de validação
    output_train = model(x_train)
    output_val = model(x_val)

    print('depois de treinar:')
    print(output_train)
    print(y_train)

    # computar a perda de treinamento e validação
    #https://www.fulljoin.com.br/posts/2020-04-27-usando-o-pytorch-no-r-treinando-o-seefood/
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # calculando os pesos atualizados de todos os parâmetros do modelo
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

    if epoch%2 == 0:
        # imprimindo a perda de validação
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

#https://nestedsoftware.com/2019/09/09/pytorch-image-recognition-with-convolutional-networks-4k17.159805.html


# Rede Neural Convolucional
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.fc1 = nn.Linear(7*12*40, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        print('Forward')
        print(x.shape)
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
    
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x,1) #x.view(-1, 7*12*40)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#Do conjunto de dados 70% é treinamento e 30% é teste
# Carregando o conjunto de dados
#arquivo de contém o id de cada imagem e seu rótulo correspondente
#id, direção, aceleração
train = pd.read_csv('gravacoes/video2fr_teste.csv')
#arquivo contém apenas os ids e temos que prever seus rótulos correspondentes
valida_csv = pd.read_csv('gravacoes/video2fr_teste.csv')
#arquivo de envio de amostra nos dirá o formato em que devemos enviar as previsões
sample_submission = pd.read_csv('gravacoes/resultados.csv')

train.head()

#Vamos ler todas as imagens uma por uma e empilhá-las uma sobre a outra em um array
# loading training images
#https://github.com/lapisco/LapiscoTraining/blob/master/Questions/3/answer.py

i=0
train_img = []
for img_name in tqdm(train['id']):    
    image_path = 'dados_treino/' + str(img_name) + '.jpg'
    image = Image.open(image_path)
    image = image.resize((40, 60))
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    pixels /= 255.0    #(normalizar entre -1 e 1)
    train_img.append(pixels)
    if i < 99:
        i += 1
    else:
        break
   
# converting the list to numpy array
train_x = np.array(train_img)

# defining the target
train_y = train['dir'].values
train_y = train_y[0:100]

# visualizando images
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(train_x[i])
plt.subplot(222), plt.imshow(train_x[i+25])
plt.subplot(223), plt.imshow(train_x[i+50])
plt.subplot(224), plt.imshow(train_x[i+75])
#plt.show()

# criação de um conjunto de validação
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.5)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)


# converting training images into torch format
train_x = train_x.reshape(train_x.shape[0], 3, 40, 60)
#train_x = train_x.reshape(736, 720000)
train_x  = torch.from_numpy(train_x)
# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)
# shape of training data
train_x.shape, train_y.shape
# converter as imagens de validação
#val_x = val_x.reshape(736, 720000)
val_x = val_x.reshape(val_x.shape[0], 3, 40, 60)
val_x  = torch.from_numpy(val_x)
# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)
# shape of validation data
val_x.shape, val_y.shape


#chamar este modelo e definir o otimizador e a função de perda para o modelo
# defining the model
model = Net()
print(model)

#parâmetros de aprendizagem de um modelo são retornados: 
params = list(model.parameters())

# definindo a função de perda
criterion = nn.CrossEntropyLoss()

# definindo o otimizador
#optimizer = Adam(model.parameters(), lr=0.07)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []

# treinar o modelo
for epoch in range(n_epochs):
    treinando(epoch)

# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
#plt.show()

# previsão para conjunto de treinamento
with torch.no_grad():
    output = model(train_x.cuda())
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# precisão no conjunto de treinamento
accuracy_score(train_y, predictions)

# prediction for validation set
with torch.no_grad():
    output = model(val_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on validation set
accuracy_score(val_y, predictions)

#Gerando previsões para o conjunto de teste
# loading test images

test_img = []
for img_name in tqdm(test['id']):
    image_path = 'dados_validados/' + str(img_name) + '.jpg'
    image = Image.open(image_path)
    pixels = asarray(image)
    print('Data Type: %s' % pixels.dtype)
    #print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    pixels = pixels.astype('float32')
    pixels /= 255.0
    test_img.append(pixels)
    #print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    '''
    # defining the image path
    image_path = 'gravacao/video2fr/' + str(img_name) + '.jpg'
    # reading the image
    img = imread(image_path, as_gray=False)
    # normalizing the pixel values
    #img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    test_img.append(img)
    '''


# convertendo a lista em array numpy
test_x = np.array(test_img)
test_x.shape

#converter imagens de treinamento em formato torch
test_x = test_x.reshape(1000, 1, 64, 64)
test_x  = torch.from_numpy(test_x)
test_x.shape

#previsões para o conjunto de teste

# gerando previsões para o conjunto de teste
with torch.no_grad():
    output = model(test_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# substituindo o rótulo com previsão
sample_submission['dir'] = predictions
sample_submission.head()

# https://sigmoidal.ai/hdf5-armazenamento-para-deep-learning/