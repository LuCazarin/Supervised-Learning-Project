import sys                # Para redirecionar a saída para output/output.txt
import time               # Para medir o tempo de treinamento e teste
import warnings           # Para capturar warnings de convergência
from pathlib import Path  # Para manipular caminhos de arquivos

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                             # Para plotar gráficos em arquivos, sem mostrar na tela
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning # Para capturar warnings de convergência

from sklearn.datasets import load_wine, fetch_openml                 # Para carregar os datasets
from sklearn.model_selection import train_test_split, ParameterGrid  # Para dividir os dados em treino e teste, e para gerar as combinações de hiperparâmetros
from sklearn.preprocessing import StandardScaler                     # Para padronizar os dados
from sklearn.neural_network import MLPClassifier                     # Para criar o modelo de Rede Neural
from sklearn.neighbors import KNeighborsClassifier                   # Para criar o modelo de k-NN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

seed = 42 # Para garantir a repetibilidade dos resultados

def executar_experimento(nome_base, X, y, modelo_base, grid_parametros):
    """Divide os dados, aplica padronização e testa todas as combinações de hiperparâmetros."""

    # Divisão 70% Treino / 30% Teste. O random_state é a seed que aleatoriza o dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    # Padronização (Obrigatório para MLP e k-NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # fit: calcula média e desvio padrão nos dados de treino; transform: padroniza os dados para ter média=0 e desvio padrão=1
    X_test_scaled = scaler.transform(X_test) # transforma os dados de teste usando os parâmetros calculados no treino

    resultados = [] # Lista para armazenar os resultados de cada combinação de hiperparâmetros
    matrizes = {}   # Dicionário para armazenar as matrizes de confusão de cada combinação de hiperparâmetros

    combinacoes = list(ParameterGrid(grid_parametros)) # Gera todas as combinações de hiperparâmetros

    for idx, params in enumerate(combinacoes):
        modelo = modelo_base.set_params(**params)
        nome_config = f"Config_{idx+1}"

        # Mede o tempo de Treinamento e diferencia se o modelo convergiu ou não
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            t_inicio_treino = time.time()
            modelo.fit(X_train_scaled, y_train)
            t_treino = time.time() - t_inicio_treino
            convergiu = not any(issubclass(aviso.category, ConvergenceWarning) for aviso in avisos)

        n_iter = getattr(modelo, "n_iter_", None)
        if not convergiu:
            print(f"  Não convergiu: {nome_base} {nome_config} {params} (n_iter={n_iter})")

        # Mede o tempo de Classificação (Inferência)
        t_inicio_teste = time.time()
        y_pred = modelo.predict(X_test_scaled)
        t_teste = time.time() - t_inicio_teste

        # Calcula as métricas (average='macro' atende bases multiclasse e binárias)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        # Armazenar Resultados
        matrizes[nome_config] = {'cm': cm, 'params': params}

        resultados.append({
            'Base': nome_base,
            'Configuração': nome_config,
            'Parâmetros': str(params),
            'Convergiu': convergiu,
            'n_iter': n_iter,
            'Tempo Treino (s)': t_treino,
            'Tempo Teste (s)': t_teste,
            'Acurácia': acc,
            'Precisão': prec,
            'Recall': rec,
            'F1-Score': f1
        })

    return pd.DataFrame(resultados), matrizes

def plotar_matriz_confusao(cm, params, titulo, caminho):
    """Salva a matriz de confusão em um arquivo PNG."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{titulo}\n{params}", fontsize=10)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close()

def _nome_feature_em_linhas(nome, largura=16):
    """Quebra nomes longos de features para caber nos eixos do pairplot."""
    texto = nome.replace("_", " ")
    palavras = texto.split()
    linhas = []
    atual = ""
    for palavra in palavras:
        candidato = palavra if not atual else f"{atual} {palavra}"
        if len(candidato) <= largura:
            atual = candidato
        else:
            if atual:
                linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return "\n".join(linhas)


def plotar_dispersao_features(X, y, titulo, caminho):
    """Gera gráficos de dispersão combinando todas as features 2 a 2."""
    df_plot = X.copy()
    df_plot = df_plot.rename(columns={col: _nome_feature_em_linhas(col) for col in df_plot.columns})
    # Converte o y para string para que o Seaborn trate sempre como categoria discreta nas cores
    df_plot['Target'] = y.astype(str)

    # pairplot cruza todas as colunas numéricas; corner=True oculta os gráficos espelhados
    g = sns.pairplot(df_plot, hue='Target', corner=True, diag_kind='hist', height=1.7)
    g.fig.suptitle(titulo, y=1.01, fontsize=14)
    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    g.fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.12, wspace=0.12)
    g.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close(g.fig)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
log_path = OUTPUT_DIR / "output.txt"
log_file = open(log_path, "w", encoding="utf-8", buffering=1)
sys.stdout = log_file
print(f"Escrevendo resultados em {log_path.resolve()}", file=sys.__stderr__)

# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================
print("Carregando bases de dados...")

q_exemplos = 5

# Base Wine (Scikit-Learn)
wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = wine.target
print("\n--- Exemplos aleatórios do dataset Wine ---")
print(X_wine.assign(target=y_wine).sample(q_exemplos, random_state=seed))

# Base Phoneme (OpenML)
phoneme = fetch_openml(name='phoneme', version=1, as_frame=True, parser='auto')
X_phoneme = phoneme.data
y_phoneme = phoneme.target
print("\n--- Exemplos aleatórios do dataset Phoneme ---")
print(X_phoneme.assign(target=y_phoneme).sample(q_exemplos, random_state=seed))

# ==========================================
# 2. DEFINIÇÃO DOS HIPERPARÂMETROS
# ==========================================
# Rede Neural
mlp_base = MLPClassifier(max_iter=1000, random_state=seed)
mlp_grid = {
    'hidden_layer_sizes': [(100,), (50, 50)],   # 1 camada de 100 neurônios vs 2 camadas de 50 neurônios
    'activation': ['relu', 'tanh', 'logistic'], # Função de ativação para os neurônios
    'solver': ['adam', 'lbfgs', 'sgd']          # Algoritmo de otimização para atualizar os pesos
}

# k-NN
knn_base = KNeighborsClassifier()
knn_grid = {
    'n_neighbors': [3, 5, 7, 11],        # Número de vizinhos mais próximos a considerar
    'weights': ['uniform', 'distance'],  # Peso dos vizinhos (uniforme ou recompensar os mais próximos)
    'metric': ['euclidean', 'manhattan'] # Como medir a distância entre os vizinhos
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

nao_convergiu = relatorio_final[relatorio_final['Convergiu'] == False]
print("\n=== COMBINAÇÕES QUE NÃO CONVERGIRAM ===")
if nao_convergiu.empty:
    print("Todas as configurações convergiram.")
else:
    print(nao_convergiu[['Base', 'Configuração', 'Parâmetros', 'n_iter', 'F1-Score']].to_string(index=False))

# ==========================================
# 4. GERAÇÃO DOS GRÁFICOS
# ==========================================
print("\nGerando os gráficos em", OUTPUT_DIR.resolve())

# Dispersão 2 a 2 (Pairplot)
plotar_dispersao_features(
    X_wine, y_wine, "Dispersão das Features - Wine",
    OUTPUT_DIR / "dispersao_features_wine.png"
)
plotar_dispersao_features(
    X_phoneme, y_phoneme, "Dispersão das Features - Phoneme",
    OUTPUT_DIR / "dispersao_features_phoneme.png"
)

# Matriz de Confusão: Melhor do Wine (Considerando todos os modelos)
res_wine_todos = pd.concat([res_wine_mlp, res_wine_knn], ignore_index=True)
melhor_config_wine = res_wine_todos.loc[res_wine_todos['F1-Score'].idxmax()]
id_melhor_w = melhor_config_wine['Configuração']
base_melhor_w = melhor_config_wine['Base']
cm_dict_w = cm_wine_mlp if "MLP" in base_melhor_w else cm_wine_knn # Busca no dict certo
plotar_matriz_confusao(
    cm=cm_dict_w[id_melhor_w]['cm'],
    params=cm_dict_w[id_melhor_w]['params'],
    titulo=f"Melhor Modelo ({base_melhor_w})",
    caminho=OUTPUT_DIR / "matriz_confusao_wine.png"
)

# Matriz de Confusão: Melhor do Phoneme (Considerando todos os modelos)
res_phoneme_todos = pd.concat([res_phoneme_mlp, res_phoneme_knn], ignore_index=True)
melhor_config_phoneme = res_phoneme_todos.loc[res_phoneme_todos['F1-Score'].idxmax()]
id_melhor_p = melhor_config_phoneme['Configuração']
base_melhor_p = melhor_config_phoneme['Base']
cm_dict_p = cm_phoneme_mlp if "MLP" in base_melhor_p else cm_phoneme_knn # Busca no dict certo
plotar_matriz_confusao(
    cm=cm_dict_p[id_melhor_p]['cm'],
    params=cm_dict_p[id_melhor_p]['params'],
    titulo=f"Melhor Modelo ({base_melhor_p})",
    caminho=OUTPUT_DIR / "matriz_confusao_phoneme.png"
)

log_file.close()
