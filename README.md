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
