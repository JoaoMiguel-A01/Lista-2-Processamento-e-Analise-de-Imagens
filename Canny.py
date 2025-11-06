import cv2
import numpy as np

# Carrega a imagem
# Substitua 'imgCarro_Km.png' pelo caminho correto da sua imagem, se necessário
img = cv2.imread('imgCarro_Km.png')

# Verifica se a imagem foi carregada com sucesso
if img is None:
    print("Erro: não foi possível carregar a imagem.")
else:
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplica blur Gaussiano para suavizar a imagem e reduzir ruído
    # Tamanho do kernel (5,5) e desvio padrão 0 são escolhas comuns
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Executa a detecção de bordas Canny
    # Os dois valores de limiar são importantes para determinar quais bordas são detectadas
    # Experimente valores diferentes (por exemplo, 50,150 ou 100,200) conforme a imagem
    edges = cv2.Canny(blurred, 50, 150)

    # Exibe a imagem original e a imagem com bordas detectadas (Windows/VS Code)
    cv2.imshow('Original', img)
    cv2.imshow('Bordas - Canny', edges)

    # Aguarda uma tecla e fecha as janelas abertas
    cv2.waitKey(0)
    cv2.destroyAllWindows()