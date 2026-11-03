# Observations

For \(K\) classes, `confusion_matrix` is a \(K \times K\) table: row = true class, column = predicted class.

For **one class** \(c\) (one-vs-rest):

- \(\mathrm{TP}_c\): predicted \(c\), is \(c\)
- \(\mathrm{FP}_c\): predicted \(c\), is not \(c\)
- \(\mathrm{FN}_c\): is \(c\), predicted something else
- \(\mathrm{TN}_c\): not \(c\), not predicted \(c\)

Then:

**Accuracy** (global, not macro):

\[
\mathrm{Acc} = \frac{\sum_c \mathrm{TP}_c}{N} = \frac{\text{correct predictions}}{\text{all samples}}
\]

In the binary 2×2 view that is \((\mathrm{TP}+\mathrm{TN})/(\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN})\).

**Precision / recall / F1** use `average='macro'`: compute per class, then **unweighted mean**. That works for Wine (3 classes) and Phoneme (2 classes).

\[
P_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c},\quad
R_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c},\quad
F1_c = \frac{2 P_c R_c}{P_c + R_c}
\]

\[
P_{\mathrm{macro}} = \frac{1}{K}\sum_c P_c,\quad
R_{\mathrm{macro}} = \frac{1}{K}\sum_c R_c,\quad
F1_{\mathrm{macro}} = \frac{1}{K}\sum_c F1_c
\]

`zero_division=0` on precision: if a class is never predicted, \(P_c = 0\) instead of an error. That is why Wine Config_18 can show precision \(\approx 0.13\) and recall \(1/3\): it basically predicts one class.

Macro F1 is **not** \(2PR/(P+R)\) of the already-averaged P and R; sklearn averages the per-class F1s.

> De fato, cada máquina deveria resultar em métricas de precisão, recall e F1 para **cada** classe (no Wine, que tem 3 classes, seriam 9 métricas por máquina; no Phoneme, seriam 6 métricas por máquina), então o script faz uma média de tudo para poupar espaço.
> Chequei com a `matriz_confusão_phoneme.pgn`, correspondente à entrada de Id 61 na tabela `output.txt` e obtive:
> \(F1_0 = 0.9241320188598371\)
> \(F1_1 = 0.8057080131723381\)
> \(F1_{macro} = (F1_0 + F1_1) / 2 = 0.8649200160160876\)
> Note que \(F1_{macro}\) deu igual ao F1-Score da tabela.

---

**Wine:** target (0, 1, 2) corresponds to (Barolo, Grignolino, Barbera)

**Phoneme:** target (1, 2) corresponds to (nasal, oral)

---

A função Sigmoide e o Gradient Descent formam uma combinação problemática no aprendizado de máquina devido ao fenômeno do desvanecimento do gradiente (vanishing gradient).

**1. Saturação nos extremos**

A função Sigmoide esmaga qualquer valor de entrada para um intervalo entre 0 e 1.
• Quando as entradas são muito grandes (positivas) ou muito pequenas (negativas), a curva da Sigmoide fica completamente plana (saturada).
• Em termos matemáticos, a derivada (o gradiente) da Sigmoide nesses extremos se aproxima de zero.

**2. Gradientes muito pequenos**

A derivada máxima da função Sigmoide é de apenas 0,25 (quando a entrada é zero). Conforme você se afasta do centro, esse valor cai drasticamente.

**3. O efeito dominó no Gradient Descent**

O Gradient Descent atualiza os pesos da rede neural multiplicando os gradientes ao longo das camadas (Regra da Cadeia).
• Se você multiplica vários valores menores que 0,25 (das funções Sigmoide de cada camada), o gradiente final fica infinitamente pequeno.
• Como o gradiente dita o "tamanho do passo" na atualização dos pesos, um gradiente próximo de zero significa que a rede para de aprender (os pesos param de mudar).

## TO-DOs

- [ ] Check this info: target (0, 1, 2) corresponds to (Barolo, Grignolino, Barbera)
- [ ] Check this info: target (1, 2) corresponds to (nasal, oral)

## O que escrever

- Escolha dos datasets:
  - Abalone (descartado pq tinha dado nominal e precisava de pré-processamento antes de rodar o algoritmo)
  - Wine (dados só numéricos)
  - Phoneme (dados só numéricos)

- output
  - Convergiu = gradiente ficou "plano" antes das 1000 iter; o algoritmo não sabia pra onde ir e terminou
  - Não convergiu = gradiente ainda apontava direção de melhoria, mas bateu 1000 iter; algoritmo sabia pra onde ir mas foi parado
  - Convergir / divergir não implica resposta boa ou ruim (pesquisar "solution space" no google imagens pra fazer paralelo)
  - MLP ganhou no Wine e k-NN ganhou no Phoneme
  - MLP foi a melhor precisão geral e ao mesmo tempo a pior precisão geral, graças à configuração de hiperparâmetros
    - A pior configuração combinou "sigmoide+2 camadas+gradiente descendente" e ficou ruim demais

- Imagens
  - Algumas combinações de features podem facilitar a separação das classes, assim como mostrado na base Íris com a pétala/sépala
    - e.g. do Wine: flavonoids x proline; alcohol x flavonoids; flavonoids x color intensity
    - No Phoneme ta tudo junto e misturado -> talvez por isso ninguém acertou 100%, pq fica difícil separar
    - Matriz de confusão e diferença de métrica de classe e 'macro' (ver `README.md`)

## Versões Gabriel

```bash
gabriel:~/Desktop/IA/Supervised-Learning-Project$pip show sys time warnings pathlib pandas numpy matplotlib seaborn sklearn
WARNING: Package(s) not found: pathlib, sklearn, sys, time, warnings
Name: pandas
Version: 2.0.3
Summary: Powerful data structures for data analysis, time series, and statistics
Home-page: None
Author: None
Author-email: The Pandas Development Team <pandas-dev@python.org>
License: BSD 3-Clause License
        
        Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
        All rights reserved.
        
        Copyright (c) 2011-2023, Open source contributors.
        
        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:
        
        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.
        
        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.
        
        * Neither the name of the copyright holder nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.
        
        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        
Location: /home/gabriel/.local/lib/python3.8/site-packages
Requires: numpy, python-dateutil, tzdata, pytz
Required-by: seaborn
---
Name: numpy
Version: 1.24.4
Summary: Fundamental package for array computing in Python
Home-page: https://www.numpy.org
Author: Travis E. Oliphant et al.
Author-email: None
License: BSD-3-Clause
Location: /home/gabriel/.local/lib/python3.8/site-packages
Requires:
Required-by: seaborn, scipy, scikit-learn, pyrr, pandas, moderngl-window, matplotlib, mapbox-earcut, manim, isosurfaces, contourpy
---
Name: matplotlib
Version: 3.7.5
Summary: Python plotting package
Home-page: https://matplotlib.org
Author: John D. Hunter, Michael Droettboom
Author-email: matplotlib-users@python.org
License: PSF
Location: /home/gabriel/.local/lib/python3.8/site-packages
Requires: pillow, importlib-resources, kiwisolver, cycler, packaging, contourpy, fonttools, python-dateutil, pyparsing, numpy
Required-by: seaborn
---
Name: seaborn
Version: 0.13.2
Summary: Statistical data visualization
Home-page: None
Author: None
Author-email: Michael Waskom <mwaskom@gmail.com>
License: None
Location: /home/gabriel/.local/lib/python3.8/site-packages
Requires: pandas, matplotlib, numpy
Required-by: 
```