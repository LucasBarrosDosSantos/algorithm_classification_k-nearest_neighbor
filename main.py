# definindo dataframe
df = []
import math


# lendo csv com tamanho das colunas
with open("breast_cancer.csv", "r") as dataset:
    for instancia in dataset.readlines():
        x = instancia.replace('\n', '').replace('N', '1').replace('O', '0').split(',')
        df.append(
            [
                float(x[0]),
                float(x[1]),
                float(x[2]),
                float(x[3]),
                float(x[4]),
                float(x[5]),
                float(x[6]),
                float(x[7])
            ]

        )

# function que retorna informações condicionadas ao parametro info
def info_dataset(amostras, info=True):
    outpu1, output2 = 0, 0

    for amostra in amostras:
        if amostra[-1] == 1:
            outpu1 += 1
        else:
            output2 += 1
    if info == True:
        print('Total de amostras :', len(amostras))
        print('Total normal : ', outpu1)
        print('Total Alterado: ', output2)

    return [len(amostras), outpu1, output2]


# qual percentual do dataset vai ser utilizado para treino ?
porcentagem = 0.5


# pegando as info do dataset
_, output1, output2 = info_dataset(df, info=False)

# definindo array para treinamento
treinamento = []
# array de testes
teste = []

max_output1 = int(porcentagem * output1)
max_output2 = int(porcentagem * output2)

total_output1 = 0
total_output2 = 0


# calcular a distancia entre os pontos
def distancia_euclidiana(p1, p2):
    dimensao = len(p1)
    soma = 0
    for i in range(dimensao):
        soma += (p1[i] - p2[i]) ** 2
    return math.sqrt(soma)


for amostra in df:
    if (total_output1 + total_output2) < (max_output1 + max_output2):
        # numero de treinamentos ainda não está satisfeito
        treinamento.append(amostra)
        if amostra[-1] == 1 and total_output1 < max_output1:
            total_output1 += 1
        else:
            total_output2 += 1
    else:
        # define array total de testes
        teste.append(amostra)

# função que executa classificação utilizando a função euclidiana
def knn(treinamento, nova_amostra, k):
    distancias = {}
    tamanho_treino = len(treinamento)

    for i in range(tamanho_treino):
        d = distancia_euclidiana(treinamento[i], nova_amostra)
        distancias[i] = d

    k_vizinhos = sorted(distancias, key=distancias.get)[:k]

    qtd_output1 = 0
    qtd_output2 = 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1:
            qtd_output1 += 1
        else:
            qtd_output2 += 1

    if qtd_output1 > qtd_output2:
        return 1
    else:
        return 0


acertos = 0
k = 9

for amostra in teste:
    classe = knn(treinamento, amostra, k)
    if amostra[-1] == classe:
        acertos += 1

print("Total de treinamento ", len(treinamento))
print("Total de testes ", len(teste))
print("Total de acertos ", acertos)
print("Porcentagem de acerto ", 100 * acertos / len(teste))
