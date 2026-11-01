import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.datasets import load_wine, fetch_openml
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Suprime os avisos de não-convergência para manter a saída limpa
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def executar_experimento(nome_base, X, y, modelo_base, grid_parametros):
    """Divide os dados, aplica padronização e testa todas as combinações de hiperparâmetros."""

    # Divisão 70% Treino / 30% Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Padronização (Obrigatório para MLP e k-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    resultados = []
    matrizes = {}

    combinacoes = list(ParameterGrid(grid_parametros))

    for idx, params in enumerate(combinacoes):
        modelo = modelo_base.set_params(**params)

        # Medição do tempo de Treinamento
        t_inicio_treino = time.time()
        modelo.fit(X_train_scaled, y_train)
        t_treino = time.time() - t_inicio_treino

        # Medição do tempo de Classificação (Inferência)
        t_inicio_teste = time.time()
        y_pred = modelo.predict(X_test_scaled)
        t_teste = time.time() - t_inicio_teste

        # Cálculo de Métricas (average='macro' atende bases multiclasse e binárias)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Armazenar Resultados
        nome_config = f"Config_{idx+1}"
        matrizes[nome_config] = {'cm': cm, 'params': params}

        resultados.append({
            'Base': nome_base,
            'Configuração': nome_config,
            'Parâmetros': str(params),
            'Tempo Treino (s)': t_treino,
            'Tempo Teste (s)': t_teste,
            'Acurácia': acc,
            'Precisão': prec,
            'Recall': rec,
            'F1-Score': f1
        })

    return pd.DataFrame(resultados), matrizes

def plotar_matriz_confusao(cm, params, titulo):
    """Plota a matriz de confusão usando seaborn."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{titulo}\n{params}", fontsize=10)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.show()

# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================
print("Carregando bases de dados...")

q_exemplos = 5

# Base Wine (Scikit-Learn)
wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = wine.target
print("\n--- Exemplos Aleatórios do dataset Wine ---")
print(X_wine.assign(target=y_wine).sample(q_exemplos))

# Base Phoneme (OpenML)
phoneme = fetch_openml(name='phoneme', version=1, as_frame=True, parser='auto')
X_phoneme = phoneme.data
y_phoneme = phoneme.target
print("\n--- Exemplos Aleatórios do dataset Phoneme ---")
print(X_phoneme.assign(target=y_phoneme).sample(q_exemplos))

# ==========================================
# 2. DEFINIÇÃO DOS HIPERPARÂMETROS
# ==========================================
# Rede Neural: max_iter elevado para permitir convergência, alpha para evitar overfitting básico
mlp_base = MLPClassifier(max_iter=1000, random_state=42)
mlp_grid = {
    'hidden_layer_sizes': [(100,), (50, 50)], # 1 camada de 100 vs 2 camadas de 50
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'lbfgs', 'sgd']
}

knn_base = KNeighborsClassifier()
knn_grid = {
    'n_neighbors': [3, 5, 7, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# ==========================================
# 3. EXECUÇÃO DOS EXPERIMENTOS
# ==========================================
print("\nIniciando experimentos no Wine...")
res_wine_mlp, cm_wine_mlp = executar_experimento("Wine (MLP)", X_wine, y_wine, mlp_base, mlp_grid)
res_wine_knn, cm_wine_knn = executar_experimento("Wine (k-NN)", X_wine, y_wine, knn_base, knn_grid)

print("Iniciando experimentos no Phoneme...")
res_phoneme_mlp, cm_phoneme_mlp = executar_experimento("Phoneme (MLP)", X_phoneme, y_phoneme, mlp_base, mlp_grid)
res_phoneme_knn, cm_phoneme_knn = executar_experimento("Phoneme (k-NN)", X_phoneme, y_phoneme, knn_base, knn_grid)

# Consolida os relatórios em um único DataFrame
relatorio_final = pd.concat([res_wine_mlp, res_wine_knn, res_phoneme_mlp, res_phoneme_knn], ignore_index=True)

# Exibe as configurações ordenadas por F1-Score
print("\n=== TOP CONFIGURAÇÕES POR F1-SCORE ===")
display_df = relatorio_final.sort_values(by='F1-Score', ascending=False)
print(display_df.to_string())

# ==========================================
# 4. PLOTAGEM DAS MATRIZES DE CONFUSÃO (Exemplo)
# ==========================================
# Plotar a matriz da melhor configuração do Wine com MLP
melhor_config_wine = res_wine_mlp.loc[res_wine_mlp['F1-Score'].idxmax()]
id_melhor = melhor_config_wine['Configuração']
plotar_matriz_confusao(
    cm=cm_wine_mlp[id_melhor]['cm'],
    params=cm_wine_mlp[id_melhor]['params'],
    titulo="Melhor MLP - Wine"
)
