# 📈 Classificação de Aderência a Investimentos com Machine Learning

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever se um cliente de uma instituição financeira irá aderir ou não a um determinado investimento oferecido em uma campanha de marketing.

O projeto abrange desde a **Análise Exploratória de Dados (EDA)** até a **criação**, **avaliação** e **exportação** de modelos preditivos.

## 📋 Sobre o Projeto

Utilizando dados históricos de uma campanha de marketing, o notebook explora o comportamento dos clientes e suas características demográficas/financeiras para identificar padrões. O problema é tratado como uma **classificação binária** (Sim/Não para aderência).

### As principais etapas incluem:
1.  **Análise Exploratória:** Visualização da distribuição dos dados e correlações usando `Plotly`.
2.  **Pré-processamento:** Tratamento de variáveis categóricas (Encoding) e normalização de dados.
3.  **Modelagem:** Comparação entre diferentes algoritmos.
4.  **Avaliação:** Comparação de métricas de acurácia.
5.  **Simulação:** Teste do modelo com novos dados fictícios.

## 🛠️ Tecnologias Utilizadas

O projeto foi desenvolvido em **Python** utilizando as seguintes bibliotecas:

* **Pandas:** Manipulação e análise de dados.
* **Plotly:** Criação de gráficos interativos para análise exploratória.
* **Matplotlib:** Visualização da árvore de decisão.
* **Scikit-learn:**
    * Pré-processamento (`OneHotEncoder`, `LabelEncoder`, `MinMaxScaler`).
    * Modelos (`DummyClassifier`, `DecisionTreeClassifier`, `KNeighborsClassifier`).
    * Métricas e utilitários (`train_test_split`, `accuracy_score`).
* **Pickle:** Serialização e salvamento do modelo treinado.

## 📊 Análise dos Dados

A base de dados (`marketing_investimento.csv`) contém as seguintes variáveis explicativas:

* `idade`: Idade do cliente.
* `estado_civil`: Estado civil (casado, solteiro, divorciado).
* `escolaridade`: Nível de instrução.
* `inadimplencia`: Se o cliente possui histórico de inadimplência.
* `saldo`: Saldo na conta.
* `fez_emprestimo`: Se o cliente fez empréstimo imobiliário.
* `tempo_ult_contato`: Duração do último contato em segundos.
* `numero_contatos`: Número de contatos realizados durante a campanha.

**Target (Alvo):**
* `aderencia_investimento`: 'sim' ou 'nao'.

## 🤖 Modelos e Resultados

Foram testados três abordagens para verificar a performance preditiva:

| Modelo | Descrição | Acurácia (Aprox.) |
| :--- | :--- | :--- |
| **Dummy Classifier** | Modelo base (Baseline) que chuta a classe mais frequente. | 60.25% |
| **Árvore de Decisão** | Modelo baseado em regras de decisão (ajustado). | **71.60%** |
| **KNN (K-Nearest Neighbors)** | Baseado na proximidade dos vizinhos (com dados normalizados). | 68.76% |

> A **Árvore de Decisão** obteve o melhor desempenho nos dados de teste.

## 🚀 Como Executar

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/mm-dantas/bank-marketing-prediction]
    ```
2.  Instale as dependências necessárias:
    ```bash
    pip install pandas sklearn plotly matplotlib
    ```
3.  Execute o notebook `bank-marketing-prediction.ipynb` (ou o nome que você definiu) em seu ambiente Jupyter ou Google Colab.

## 💾 Exportação do Modelo

Ao final do projeto, o modelo vencedor (Árvore de Decisão) e o codificador (OneHotEncoder) foram exportados utilizando a biblioteca `pickle` para uso em produção:

* `modelo_onehotenc.pkl`: Transformador das variáveis categóricas.
* `modelo_arvore.pkl`: O classificador treinado.

Exemplo de uso com novos dados:
```python
import pandas as pd
import pickle

# Carregando o modelo
modelo = pd.read_pickle('modelo_arvore.pkl')
encoder = pd.read_pickle('modelo_onehotenc.pkl')

# Novos dados
novo_cliente = pd.DataFrame({
    'idade': [45],
    'estado_civil':['solteiro (a)'],
    'escolaridade':['superior'],
    # ... demais colunas
})

# Previsão
resultado = modelo.predict(encoder.transform(novo_cliente))
