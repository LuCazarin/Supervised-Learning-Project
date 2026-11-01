# Observations

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

## Output example

```
Carregando bases de dados...

--- Exemplos Aleatórios do dataset Wine ---
     alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \
155    13.17        5.19  2.32               22.0       93.0           1.74   
153    13.23        3.30  2.28               18.5       98.0           1.80   
26     13.39        1.77  2.62               16.1       93.0           2.85   
98     12.37        1.07  2.10               18.5       88.0           3.52   
91     12.00        1.51  2.42               22.0       86.0           1.45   

     flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \
155        0.63                  0.61             1.55             7.90  0.60   
153        0.83                  0.61             1.87            10.52  0.56   
26         2.94                  0.34             1.45             4.80  0.92   
98         3.75                  0.24             1.95             4.50  1.04   
91         1.25                  0.50             1.63             3.60  1.05   

     od280/od315_of_diluted_wines  proline  target  
155                          1.48    725.0       2  
153                          1.51    675.0       2  
26                           3.22   1195.0       0  
98                           2.77    660.0       1  
91                           2.65    450.0       1  

--- Exemplos Aleatórios do dataset Phoneme ---
            V1        V2        V3        V4        V5 target
1745 -0.820346 -0.932916 -0.017434  0.296262 -0.136583      2
324  -0.189601  2.394085 -1.425312  0.406458  0.519358      2
4587 -0.660591 -0.148317  1.769464 -1.380113 -0.136583      1
4828 -0.182883  0.299738  1.157003  0.329448 -0.534676      1
4894  0.872689  0.122217 -0.379252 -0.304258 -0.136583      1

Iniciando experimentos no Wine...
Iniciando experimentos no Phoneme...

=== TOP CONFIGURAÇÕES POR F1-SCORE ===
              Base Configuração                                                                     Parâmetros  Tempo Treino (s)  Tempo Teste (s)  Acurácia  Precisão    Recall  F1-Score
0       Wine (MLP)     Config_1         {'activation': 'relu', 'hidden_layer_sizes': (100,), 'solver': 'adam'}          0.269555         0.000669  1.000000  1.000000  1.000000  1.000000
1       Wine (MLP)     Config_2        {'activation': 'relu', 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'}          0.029411         0.000667  1.000000  1.000000  1.000000  1.000000
2       Wine (MLP)     Config_3          {'activation': 'relu', 'hidden_layer_sizes': (100,), 'solver': 'sgd'}          0.698761         0.000829  1.000000  1.000000  1.000000  1.000000
3       Wine (MLP)     Config_4       {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'solver': 'adam'}          0.257990         0.000630  1.000000  1.000000  1.000000  1.000000
14      Wine (MLP)    Config_15      {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'solver': 'sgd'}          0.728901         0.000455  1.000000  1.000000  1.000000  1.000000
7       Wine (MLP)     Config_8        {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'}          0.024514         0.000846  0.981481  0.982456  0.984127  0.982861
8       Wine (MLP)     Config_9          {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'solver': 'sgd'}          0.718080         0.000725  0.981481  0.982456  0.984127  0.982861
6       Wine (MLP)     Config_7         {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'solver': 'adam'}          0.464459         0.000818  0.981481  0.982456  0.984127  0.982861
12      Wine (MLP)    Config_13     {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'solver': 'adam'}          0.336470         0.000440  0.981481  0.982456  0.984127  0.982861
13      Wine (MLP)    Config_14    {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'}          0.023692         0.000601  0.981481  0.982456  0.984127  0.982861
33     Wine (k-NN)    Config_16              {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'distance'}          0.000784         0.001255  0.981481  0.982456  0.984127  0.982861
11      Wine (MLP)    Config_12        {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'solver': 'sgd'}          0.885301         0.000722  0.981481  0.979167  0.984127  0.981117
30     Wine (k-NN)    Config_13                {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'uniform'}          0.000893         0.002066  0.981481  0.979167  0.984127  0.981117
28     Wine (k-NN)    Config_11                {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'uniform'}          0.000952         0.002071  0.981481  0.979167  0.984127  0.981117
29     Wine (k-NN)    Config_12               {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}          0.000850         0.001206  0.981481  0.979167  0.984127  0.981117
31     Wine (k-NN)    Config_14               {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}          0.000843         0.001213  0.981481  0.979167  0.984127  0.981117
9       Wine (MLP)    Config_10       {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'solver': 'adam'}          0.330683         0.000730  0.981481  0.979167  0.984127  0.981117
5       Wine (MLP)     Config_6        {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'solver': 'sgd'}          0.909344         0.000695  0.981481  0.984848  0.977778  0.980754
4       Wine (MLP)     Config_5      {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'solver': 'lbfgs'}          0.049488         0.002644  0.981481  0.984848  0.977778  0.980754
32     Wine (k-NN)    Config_15               {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'uniform'}          0.000877         0.002028  0.962963  0.966667  0.968254  0.965789
15      Wine (MLP)    Config_16   {'activation': 'logistic', 'hidden_layer_sizes': (50, 50), 'solver': 'adam'}          0.394865         0.000463  0.962963  0.966583  0.961905  0.963624
10      Wine (MLP)    Config_11      {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'solver': 'lbfgs'}          0.030797         0.000935  0.962963  0.966583  0.961905  0.963624
26     Wine (k-NN)     Config_9                {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}          0.000874         0.001960  0.962963  0.961623  0.968254  0.963572
27     Wine (k-NN)    Config_10               {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}          0.001339         0.001305  0.962963  0.961623  0.968254  0.963572
25     Wine (k-NN)     Config_8              {'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'distance'}          0.000877         0.001244  0.962963  0.960784  0.968254  0.962500
24     Wine (k-NN)     Config_7               {'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'uniform'}          0.000952         0.001999  0.962963  0.960784  0.968254  0.962500
16      Wine (MLP)    Config_17  {'activation': 'logistic', 'hidden_layer_sizes': (50, 50), 'solver': 'lbfgs'}          0.043787         0.000658  0.962963  0.961905  0.961905  0.961905
18     Wine (k-NN)     Config_1                {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'uniform'}          0.001396         0.002968  0.944444  0.943240  0.952381  0.944517
23     Wine (k-NN)     Config_6               {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}          0.000858         0.001194  0.944444  0.943240  0.952381  0.944517
19     Wine (k-NN)     Config_2               {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}          0.000845         0.001325  0.944444  0.943240  0.952381  0.944517
22     Wine (k-NN)     Config_5                {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'uniform'}          0.000870         0.002024  0.944444  0.943240  0.952381  0.944517
21     Wine (k-NN)     Config_4               {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'distance'}          0.001244         0.001299  0.944444  0.944444  0.952381  0.944056
20     Wine (k-NN)     Config_3                {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}          0.001464         0.002585  0.944444  0.944444  0.952381  0.944056
61  Phoneme (k-NN)    Config_10               {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}          0.005914         0.017219  0.890875  0.875925  0.855836  0.864920
53  Phoneme (k-NN)     Config_2               {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}          0.005734         0.011653  0.889642  0.871082  0.859262  0.864824
43   Phoneme (MLP)    Config_10       {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'solver': 'adam'}         45.755529         0.011481  0.884710  0.864939  0.853315  0.858783
59  Phoneme (k-NN)     Config_8              {'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'distance'}          0.005765         0.019168  0.882861  0.865100  0.847093  0.855289
63  Phoneme (k-NN)    Config_12               {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}          0.006584         0.022732  0.882244  0.864930  0.845429  0.854236
57  Phoneme (k-NN)     Config_6               {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}          0.005637         0.015568  0.882244  0.865349  0.844815  0.854040
65  Phoneme (k-NN)    Config_14               {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}          0.005801         0.024881  0.881011  0.862556  0.845170  0.853101
52  Phoneme (k-NN)     Config_1                {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'uniform'}          0.010237         0.021509  0.878545  0.857261  0.845882  0.851231
67  Phoneme (k-NN)    Config_16              {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'distance'}          0.006076         0.029051  0.878545  0.860623  0.840354  0.849455
55  Phoneme (k-NN)     Config_4               {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'distance'}          0.006753         0.013932  0.876695  0.856169  0.841502  0.848275
60  Phoneme (k-NN)     Config_9                {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'uniform'}          0.007126         0.019284  0.876695  0.856903  0.840274  0.847874
35   Phoneme (MLP)     Config_2        {'activation': 'relu', 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'}         30.865952         0.003177  0.873613  0.850799  0.840549  0.845390
37   Phoneme (MLP)     Config_4       {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'solver': 'adam'}          9.885135         0.001580  0.869914  0.842815  0.844072  0.843439
44   Phoneme (MLP)    Config_11      {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'solver': 'lbfgs'}         31.413239         0.011275  0.868681  0.842997  0.838287  0.840578
47   Phoneme (MLP)    Config_14    {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'}         41.776572         0.006309  0.866215  0.842206  0.830400  0.835915
38   Phoneme (MLP)     Config_5      {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'solver': 'lbfgs'}         42.830825         0.005304  0.864365  0.840351  0.827249  0.833321
49   Phoneme (MLP)    Config_16   {'activation': 'logistic', 'hidden_layer_sizes': (50, 50), 'solver': 'adam'}         15.786981         0.006627  0.861899  0.834588  0.830417  0.832451
62  Phoneme (k-NN)    Config_11                {'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'uniform'}          0.005712         0.022177  0.863748  0.841371  0.823128  0.831341
40   Phoneme (MLP)     Config_7         {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'solver': 'adam'}         22.306361         0.011811  0.860049  0.831563  0.830336  0.830945
54  Phoneme (k-NN)     Config_3                {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}          0.005685         0.014443  0.861899  0.838515  0.821819  0.829393
64  Phoneme (k-NN)    Config_13                {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'uniform'}          0.005798         0.024227  0.860666  0.837953  0.818490  0.827177
50   Phoneme (MLP)    Config_17  {'activation': 'logistic', 'hidden_layer_sizes': (50, 50), 'solver': 'lbfgs'}         35.731514         0.005078  0.856967  0.828788  0.823856  0.826249
34   Phoneme (MLP)     Config_1         {'activation': 'relu', 'hidden_layer_sizes': (100,), 'solver': 'adam'}         10.752284         0.001339  0.855734  0.828721  0.818684  0.823404
56  Phoneme (k-NN)     Config_5                {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'uniform'}          0.005718         0.017740  0.857583  0.834517  0.813852  0.822996
39   Phoneme (MLP)     Config_6        {'activation': 'relu', 'hidden_layer_sizes': (50, 50), 'solver': 'sgd'}         15.200575         0.001760  0.852035  0.820783  0.824664  0.822674
46   Phoneme (MLP)    Config_13     {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'solver': 'adam'}         16.066109         0.003936  0.853884  0.825124  0.819832  0.822392
41   Phoneme (MLP)     Config_8        {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'}         73.463397         0.012762  0.847719  0.816995  0.814241  0.815593
66  Phoneme (k-NN)    Config_15               {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'uniform'}          0.005708         0.029263  0.850185  0.823802  0.807388  0.814799
45   Phoneme (MLP)    Config_12        {'activation': 'tanh', 'hidden_layer_sizes': (50, 50), 'solver': 'sgd'}         39.561590         0.010807  0.844020  0.812848  0.808552  0.810640
58  Phoneme (k-NN)     Config_7               {'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'uniform'}          0.005577         0.019929  0.845253  0.816755  0.803284  0.809460
36   Phoneme (MLP)     Config_3          {'activation': 'relu', 'hidden_layer_sizes': (100,), 'solver': 'sgd'}          5.699495         0.001532  0.794698  0.752581  0.748469  0.750449
42   Phoneme (MLP)     Config_9          {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'solver': 'sgd'}          1.887162         0.011297  0.733662  0.672053  0.643864  0.652468
48   Phoneme (MLP)    Config_15      {'activation': 'logistic', 'hidden_layer_sizes': (100,), 'solver': 'sgd'}          3.345956         0.016659  0.733662  0.671682  0.638951  0.648044
51   Phoneme (MLP)    Config_18    {'activation': 'logistic', 'hidden_layer_sizes': (50, 50), 'solver': 'sgd'}          0.344267         0.003675  0.706535  0.353268  0.500000  0.414017
17      Wine (MLP)    Config_18    {'activation': 'logistic', 'hidden_layer_sizes': (50, 50), 'solver': 'sgd'}          0.079076         0.000599  0.388889  0.129630  0.333333  0.186667
```
