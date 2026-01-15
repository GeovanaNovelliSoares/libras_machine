# 🤟 Libras Machine: Reconhecimento de Alfabeto ASL em Tempo Real

Este projeto utiliza **Visão Computacional** e **Deep Learning** para interpretar letras do alfabeto da Língua de Sinais Americana (ASL) em tempo real através da webcam. O sistema utiliza MediaPipe para o rastreamento das mãos e uma Rede Neural Convolucional (CNN) treinada em TensorFlow para a classificação.

---

## 🚀 Funcionalidades

* **Detecção Inteligente:** Usa o MediaPipe para localizar a mão e criar um recorte dinâmico (crop), garantindo que o modelo foque apenas no sinal realizado.
* **Estabilização de Texto:** Inclui um contador de frames para evitar "pulos" na detecção. Uma letra só é adicionada à palavra se for detectada consistentemente por 15 frames.
* **Pipeline de Treinamento:** Script incluso para processamento de imagens com Data Augmentation e treinamento de rede neural.

## 🛠️ Tecnologias e Arquitetura

O projeto foi construído utilizando:
* **Python 3.x**
* **TensorFlow/Keras:** Criação e treinamento da CNN.
* **OpenCV:** Manipulação de vídeo e processamento de imagem.
* **MediaPipe:** Extração de marcos (landmarks) da mão em tempo real.



[Image of a convolutional neural network architecture]


### Estrutura da Rede Neural:
1.  **3 Camadas Convolucionais:** Com filtros de 32, 64 e 128 para extração de características.
2.  **Max Pooling:** Redução de dimensionalidade espacial
