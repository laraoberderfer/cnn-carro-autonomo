# Lara Popov Zambiasi Bazzi Oberderfer
# Redes Neurais Convolucionais

# grava imagens e arquivo com rotulos
import gym
import numpy as np
import sys
import os
from pyglet.window import key
import imageio
from gera_lotes import LEFT, RIGHT, GO, ACTIONS

# limita a quantidade de imagens que grava
total_imagens = 100

def action_to_id(a):
    if all(a == [-1, 0, 0]):
        return LEFT
    elif all(a == [1, 0, 0]):
        return RIGHT
    else:
        return GO


if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.exit("Digite : python3 grava_imagens.py pasta")

    env = gym.make('CarRacing-v0').env
    env.reset()

    # pasta e arquivo de rótulos
    pasta = sys.argv[1]
    # cria uma pasta dentro dessa pasta
    imgs = os.path.join(pasta, "imagens")
    # cria o arquivo rotulos
    rotulos = os.path.join(pasta, "rotulos.txt")
    os.makedirs(imgs, exist_ok=True)
    a = np.array([0.0, 0.0, 0.0])

    # isso é do gym
    def key_press(k, mod):
        global restart
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8  # 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.reset()
    for i in range(100):
        env.step([0, 0, 0])
        env.render()

    arquivo = open(rotulos, 'w')
    img = {a: 0 for a in ACTIONS}

    i = 0
    while i < total_imagens:
        s, r, done, info = env.step(a)
        acao = action_to_id(a)
        if img[acao] < total_imagens:
            img[acao] += 1
            imageio.imwrite(os.path.join(pasta, 'imagens', 'img-%s.jpg' % i), s)
            # gravando no arquivo nome da imagem, rótulo
            arquivo.write('%s %s\n' % ('img-%s.jpg' %i, acao))
            arquivo.flush()
            i += 1
            print(img)
        env.render()
    env.close()
