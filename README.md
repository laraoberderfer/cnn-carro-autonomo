# Rede convolucional para carro autonomo
-------------------------------------------------------------------------------------------------
#### Preparando o ambiente:
Instalando dependências no linux/windows para as bibliotecas (as vezes não precisa todas)

- [x] sudo apt-get install python3-pippip3 install numpy
- [x] pip3 install matplotlib
- [x] pip3 install Box2D 
- [x] pip3 install gym
- [x] pip3 install pandas 
- [x] pip3 install pyglet
- [x] sudo apt-get install freeglut3-devsudo pip3 install scikit-imagegit clone https://github.com/scikit-image/scikit-image.git
- [x] cd scikit-image
- [x] pip3 install -e .pip3 install sklearnpip3 install tqdmpip3 install torch
-------------------------------------------------------------------------------------------------
#### Gerar treinamento:

- corrida.py: simulador disponível em: https://gym.openai.com/envs/CarRacing-v0/

- model.py: Rede neural convolucional (existem dois modelos disponíveis)

- grava_imagens.py: grava imagens e arquivo com rotulos
    Para gravar uma corrida manualmente para armazenar o lote de imagens e rótulos 
    Digite: ```python3 grava_imagens.py pasta```

- gera_lotes.py: arquivo auxiliar de gerar lotes

- treino.py: treinamento da rede neural a partir de lote de imagens
    Para treinar a rede neural convolucional (enviar a pasta)
    Digite: ```python3 treino.py treinoX``
  
- autonomo.py: arquivo que executa o carro autonomo
    Para executar o teste da rede neural (a pasta e o treino podem variar)
    Digite: ```python3 autonomo.py treinoX/treino_XX.pt```

-------------------------------------------------------------------------------------------------
#### Autora
- Professora Lara Popov Zambiasi Bazzi Oberderfer
- Docente de Informática - Câmpus Chapecó
- Acadêmica do Doutorado de Automação e Sistemas da UFSC 2021  
- Instituto Federal de Santa Catarina - Câmpus Chapecó
Rua Nereu Ramos, 3450D, Universitário, Chapecó / SC - CEP: 89813-000 
- IFSC: http://www.chapeco.ifsc.edu.br
-------------------------------------------------------------------------------------------------
