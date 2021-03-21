'''
Arquivo que Tranforma arquivo mp4 em imagens
importe a biblioteca necessária para utilizar a cv2: $ pip install opencv-python
'''
import cv2
import os
import time

def criaImagem(video):
    # lendo video do caminho especificado
    cam = cv2.VideoCapture("gravacoes/"+video+".mp4")

    try:
        # criando a pasta
        if not os.path.exists('gravacoes/'+video):
            os.makedirs('gravacoes/'+video)

    # se não conseguir criar mostrar erro
    except OSError:
        print('Erro: não foi possível criar o diretório do arquivo')

    # frame
    currentframe = 0
    contador = 0

    while (True):
        # lendo o frame da imagem
        ret, frame = cam.read()
        
        if ret:
            if contador % 1 == 0: #alterações para 10, 30 e 50 frames para print de imagem
                # se existir video contiue criando imagens
                name = './gravacoes/'+video+'/' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                # escrevendo as imagens extraidas
                cv2.imwrite(name, frame)
            
            contador+=1

            # incrementa o contador que vai mostrar qntas imagens são criadas
            currentframe += 1
        else:
            break  

    # libera todos espaços e janelas
    cam.release()
    cv2.destroyAllWindows()

#chama função para execucao
criaImagem("video2fr")  #passe aqui o nome do vídeo