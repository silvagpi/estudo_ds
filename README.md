# estudo_ds - melhor definição
a. Como foi a definição da sua estratégia de modelagem?

A partir dos dados fiz alguns testes, mas a estratégia foi avaliar as regressões mais utilizadas e verificar a que possuia maior acuracia e precisão.

b. Como foi definida a função de custo utilizada?

utilizei uma função kernel que é a que com um custo de 10 de penalidade do modelo mostrou mais adequada na classificação.

c. Qual foi o critério utilizado na seleção do modelo final?
o resultado da tabela de acuracia.
O modelo ajustados com as variaveis que mais impactam o modelo preditivo (excluindo maiores p).

d. Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Os dois modelos utilizados no final foram validados o SVM por uma base de teste e treino para excecução dos testes, porém a função sm.OLS de regressão já trouxe todos os ajustes necessários.

e. Quais evidências você possui de que seu modelo é suficientemente bom?
Sua acurácia e seu R² são extremamente significativos.
