# IMPORTANTE: Antes de começar, instale os pacotes contidos em requirements.txt
# Python -m pip install -r requirements.txt

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# IMPORTANTE: Instruções para Repositório de Dados
# 1. Crie um documento no Google 
# 2. Insira dados com estes cabeçaçhos: faltas, nota_1, nota_2, resultado
# 3. Adicione de 10 a 15 linhas conforme o exemplo abaixo:

"""
IMPORTANTE: Instruções para Repositório de Dados
1. Crie um documento no Google 
2. Insira dados com estes cabeçaçhos: faltas, nota_1, nota_2, resultado
3. Adicione de 10 a 15 linhas conforme o exemplo abaixo:
    8,2.5,4.5, reprovado
    2,5.0,7.0, aprovado
    ....(adicionar até 15 linhas)
4. Ir em "Arquivo" -> "Compartilhar" -> "Publicar na web"
5. Selecionar "Valores separados por virgula (.csv)" e clicar em "Publicar"
6. COPIAR O LINK GERADO E COLAR ABAIXO: 
"""
URL_DADOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRWh9OO1M4RYiuBWoQOsAOsSgXMyuC4inTAiauGBXPK8oRc-foExR41kazYG2mBu3DAxR6P8nWHTz53/pub?output=csv"

ARQUIVO_MODELO = 'modelo_desempenho.pkl'

def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS}...")
    # Convertendo dados do arquivo csv em estrutura de dados
    data = pd.read_csv(URL_DADOS)
    print("-- Dados Carregados --")
    print(data.head())

    print("--- Preparando dados para o treino ---")
    features = ["faltas", "nota_1", "nota_2"]
    target = "resultado"
    # X recebe as característisticas(features) e Y recebe o 
    # rótulo(chamado de alvo ou target, nesse caso)
    X = data[features]
    Y = data[target]

    # Instancia "modelo" com 3 vizinhos (n_neighbors=3) 
    modelo = KNeighborsClassifier(n_neighbors=3)
    # Para treinar o modelo usa-se o método .fit() com as entradas X e Y
    modelo.fit(X, Y)
    print(f"--- Modelo Treinado! Classes: {modelo.classes_} ---")

    # Salvando o modelo treinado no arquivo cujo nome está armazenado na 
    # constante ARQUIVO_MODELO
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"--- Modelo salvo com sucesso em '{ARQUIVO_MODELO}'")

if __name__ == "__main__":
    treinar_modelo()