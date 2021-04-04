# Lara Popov Zambiasi Bazzi Oberderfer
# Redes Neurais Convolucionais

import gym
import numpy as np
from numpy import asarray
import sys
import torch
from torch.autograd import Variable
from torchvision import transforms
import PIL
from torch.nn import Softmax
from pyglet.window import key
from gera_lotes import LEFT, RIGHT, GO, ACTIONS
from gera_lotes import transforma_imagem
from model import Net, Net2
import time


id_to_steer = {
    LEFT: -1,
    RIGHT: 1,
    GO: 0,
}

if __name__ == '__main__':

    # mensagem para o usurario
    if len(sys.argv) < 2:
        sys.exit("Digite : python3 autonomo.py pasta/treino_x.pesos")

    # carregando o modelo treinado
    treino = sys.argv[1]
    model = Net2()
    model.load_state_dict(torch.load(treino))
    model.eval()

    # carregando o gym
    env = gym.make('CarRacing-v0').env
    env.reset()

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.reset()
    total_percurso = 0.0

    # inicializacao
    for i in range(50):
        env.step([0, 0, 0])
        env.render()

    i = 0
    cadencia = 0
    atualiza = 0

    # avaliando o agente


    #timer_inicial = datetime.now()


    timer_total = time.time() + 30 # segundos
    timer_inicial = time.time()
    print(timer_total)
    total_reward = 0.0
    total_percurso = 0

    while True:
        s, r, done, info = env.step(a)
        total_reward += r

        # print(int(total_reward))

        total_percurso += 1

        img = s.copy()

        # transforma o numpy array para PIL image
        # pq precisamos uma imagem como entrada
        img = PIL.Image.fromarray(img)
        input = transforma_imagem(img) # transforma img para tensor
        input = Variable(input[None, :])
        input = model(input)
        x = Softmax(dim=1)
        output = x(input)
        acelera, index = output.max(1)
        acelera = acelera.data[0].item()
        index = index.data[0].item()

        # calibrando
        # print(id_to_steer[index])
        a = [0.0,0.0,0.0]

        if index == 0:
            a[0] = -1.0
            print("esquerda ", index)
        if index == 1:
            a[0] = 1.0
            print("direita ", index)
        if index == 2: #GO
            # diminuindo a velocidade para poder fazer curvas
            if cadencia % 2 == 0:
                a[1] = 1.0
                print("acelera " , index)
            cadencia += 1

        # print(a)
        env.render()
        if time.time() > timer_total:
            break
    #timer_fim = datetime.now()
    env.close()

    timer_total = time.time() - timer_inicial
    media_pontos_tempo = total_reward/timer_total
    print("Tempo total: ", timer_total)
    print("Total pontos: ", total_reward)
    print("Total pontos/segundos: ", media_pontos_tempo)