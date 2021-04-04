# Rede convolucional para carro autonomo

#### Preparando o ambiente:
instalando dependências no linux para as bibliotecas

- [x] unrar x corridaV2.rar
- [x] sudo apt-get install python3-pippip3 install numpy
- [x] pip3 install matplotlib
- [x] pip3 install Box2Dpip3 install gym
- [x] pip3 install pandaspip3 install pyglet
- [x] sudo apt-get install freeglut3-devsudo pip3 install scikit-imagegit clone https://github.com/scikit-image/scikit-image.git
- [x] cd scikit-image
- [x] pip3 install -e .pip3 install sklearnpip3 install tqdmpip3 install torch

#### Arquivos:

- corrida.py: gera os videos para treino manual

- video2img.py: transforma video para imagem

- criaVideo.py: gera video a partir de imagens

- criacsv.py: gera arquivo csv

- model.py: Rede convolucional

#### Gerar treinamento:
- no terminal: ```python3 corrida.py``` para iniciar o treinamento
- no terminal: ```python3 video2img.py``` para transformar o video em imagens
- no terminal: ```python3 treino.py``` para treinar a rede convolucional
- no terminal: ```python3 autonomo.py``` para ver o aprendizado


-------------------------------------------------------------------------------------------------
#### Autora
Professora Lara Popov Zambiasi Bazzi Oberderfer

Docente de Informática - Câmpus Chapecó

Instituto Federal de Santa Catarina - Câmpus Chapecó
Rua Nereu Ramos, 3450D, Universitário, Chapecó / SC - CEP: 89813-000

http://www.chapeco.ifsc.edu.br
-------------------------------------------------------------------------------------------------