import numpy as np
import cv2

# Lê a imagem de entrada
img = cv2.imread('imgTeste.png')  # Substitua pelo caminho do seu arquivo, se necessário

# Verifica se a imagem foi carregada com sucesso
if img is None:
    print("Erro: não foi possível carregar a imagem.")
else:
    # Reorganiza a imagem em um array 2D de pixels com 3 canais de cor (B, G, R)
    # Cada linha representa um pixel, colunas representam os valores B, G, R
    Z = img.reshape((-1, 3))

    # Converte para np.float32 para o algoritmo K-Means
    Z = np.float32(Z)

    # Define o critério de parada do K-Means
    # (tipo, max_iter, epsilon)
    # cv2.TERM_CRITERIA_EPS: para quando a precisão epsilon for alcançada
    # cv2.TERM_CRITERIA_MAX_ITER: para quando o número máximo de iterações for atingido
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Número de clusters (K)
    K = 8  # Ajuste este valor para controlar o número de cores

    # Aplica o K-Means
    # ret: medida de compacidade (soma dos quadrados das distâncias)
    # label: rótulos de cada pixel indicando o cluster
    # center: centros dos clusters (novos valores de cor)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converte os centros de volta para uint8
    center = np.uint8(center)

    # Mapeia os rótulos para os novos valores de cor dos centros
    res = center[label.flatten()]

    # Redimensiona o resultado de volta para as dimensões originais da imagem
    res2 = res.reshape((img.shape))

    # Exibe a imagem original e a imagem quantizada (Windows/VS Code)
    cv2.imshow('Original', img)
    cv2.imshow('K-Means - Quantizada', res2)

    # Aguarda uma tecla e fecha as janelas abertas
    cv2.waitKey(0)
    cv2.destroyAllWindows()