# Trabalho 1: Classificação Supervisionada (MLP vs. k-NN)

*Projeto acadêmico desenvolvido para a disciplina de Inteligência Artificial — Universidade Federal do ABC (UFABC).*  
*Autores: Gabriel Angelo Sembenelli (RA: 11202022168) e Luana Andreza Franca Cazarin (RA: 11202131377).*

### Pré-requisitos
- **Python versão 3.8.10** (versão oficial utilizada nos testes e validação do ambiente de desenvolvimento pelo grupo)
- Sistema operacional Linux, macOS ou Windows com terminal compatível (Bash ou PowerShell)

### Configuração do Ambiente
Para garantir a exata reprodutibilidade dos experimentos e evitar conflitos de dependências:

```bash
# Cria o ambiente virtual
python3 -m venv .venv

# Ativa o ambiente virtual (Linux / macOS)
source .venv/bin/activate

# Ativa o ambiente virtual (Windows PowerShell)
# .venv\Scripts\Activate.ps1

# Instala as dependências estritas
pip install --upgrade pip
pip install -r requirements.txt
```

#### Especificações de Versões (`requirements.txt`)
Localmente, o projeto foi testado e validado nas seguintes versões de bibliotecas:
- `numpy==1.24.4`
- `pandas==2.0.3`
- `matplotlib==3.7.5`
- `seaborn==0.13.2`
- `scikit-learn==1.1.3`

### Execução

```bash
python3 projeto_completo.py
```

### Opção alternativa

O script também foi testado no ambiente virtual do Google Colab.

### Artefatos Gerados
Ao concluir a execução, o script gera e salva automaticamente no diretório local (`output/`):
- **`output.txt`**: Relatório em texto contendo os tempos de execução (treino e teste), status de convergência de épocas, acurácia global, precisão macro, *recall* macro e $F_1$-*score* macro de todas as configurações testadas.
- **Gráficos de Dispersão (`.png`)**: Visualizações bidimensionais das características dos datasets (*Wine* e *Phoneme*).
- **Matrizes de Confusão (`.png`)**: Mapeamentos visuais dos melhores modelos obtidos para cada dataset.
