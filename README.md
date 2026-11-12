# Recomendação de filmes com filtragem colaborativa (KNN) no MovieLens 100K

Quando alguém avalia um filme, ela diz algo sobre o próprio gosto. Se outras pessoas deram notas parecidas aos mesmos filmes, as notas que elas deram a filmes que você ainda não viu são uma boa pista do que você acharia deles. Este projeto testa essa ideia de forma controlada: um sistema de recomendação **baseado em usuários** (K vizinhos mais próximos), avaliado no conjunto **MovieLens 100K**, com um artigo em formato IEEE que documenta tudo.

A pergunta que guiou o trabalho foi simples: **dá para prever, com notas de usuários parecidos, quanto alguém gostaria de um filme, e isso é melhor do que simplesmente usar a média?** Para respondê-la, formulamos quatro hipóteses antes de rodar qualquer experimento:

| | Hipótese | Resultado |
|---|---|---|
| H1 | A medida de similaridade (cosseno, Pearson ou MSD) altera o erro, e Pearson é a melhor | **Refutada.** MSD empatou com Pearson e foi melhor no modelo básico |
| H2 | Descontar a média de cada usuário (notas centralizadas) reduz o erro | **Confirmada**, em todas as 5 partições |
| H3 | O KNN erra menos que as médias (global, do usuário, do filme) | **Confirmada**, com vantagem pequena |
| H4 | O erro cai com o histórico do usuário, mas listas de 10 filmes sofrem com a esparsidade | **Parcialmente confirmada** |

Hipótese refutada também é resultado, e o artigo trata as quatro com o mesmo peso.

---

## O que foi comparado, e por quê

Todos os modelos têm a mesma base: para prever a nota do usuário *u* ao filme *i*, procuramos os *k* usuários mais parecidos com *u* que avaliaram *i* e combinamos as notas deles. O que muda é o que "parecido" significa e como as notas são combinadas.

- **Cosseno** trata cada usuário como um vetor de notas. É rápido e é o ponto de partida mais comum, mas ignora que há quem dê 5 a tudo e quem nunca passe de 3.
- **Correlação de Pearson**, calculada só sobre os filmes que os dois avaliaram, corrige exatamente esse viés de escala. Por isso era a nossa aposta inicial.
- **MSD** (1 / (1 + diferença quadrática média)) mede quão próximas são as notas em valor absoluto. Foi incluída justamente para testar se a sofisticação do Pearson compensa.
- **Previsão básica versus centralizada**: a básica faz a média ponderada das notas dos vizinhos; a centralizada pesa o *desvio* de cada vizinho em relação à própria média e soma à média do usuário-alvo. É a variação que mais pesou nos resultados.
- **Médias (global, do usuário, do filme)** funcionam como régua. Um sistema só merece existir se ganhar delas, e a média por filme é um adversário surpreendentemente duro neste conjunto.

Escolhemos o KNN porque o resultado é interpretável: dá para abrir uma previsão e ver quais vizinhos a produziram. Modelos de fatoração de matriz ficaram fora do escopo (e aparecem como passo seguinte natural).

## Os dados

O MovieLens 100K reúne **100.000 avaliações** de **943 usuários** sobre **1.682 filmes**, coletadas em 1997–1998, em notas de 1 a 5. A matriz usuário × filme tem apenas **6,3% de células preenchidas**, e essa esparsidade explica quase tudo o que acontece depois. A nota média é 3,53, a nota mais comum é 4 (34%) e cada usuário avaliou ao menos 20 filmes (mediana: 65). A distribuição é concentrada: **os 20% de filmes mais avaliados recebem 65% das notas**, e 32% dos filmes têm menos de 10 avaliações.

Os arquivos de dados **não estão neste repositório**: a licença do MovieLens exige autorização para redistribuir. Baixe em <https://grouplens.org/datasets/movielens/100k/> e cite Harper e Konstan (2015).

## Como os números foram obtidos

Usamos as 5 partições oficiais (u1 a u5, 80% treino / 20% teste). O valor de *k* (entre 5 e 100) e a configuração vencedora foram escolhidos em uma **validação interna de 10% do próprio treino**, de modo que o conjunto de teste nunca influenciou nenhuma decisão. Quando um par de usuários tem menos de 5 filmes em comum, a similaridade não é usada; se não sobrar vizinho, a previsão cai para a média do usuário, e toda previsão é limitada ao intervalo [1, 5]. A semente é fixa (42) e a execução completa leva cerca de 2,5 minutos.

Além do erro de previsão (**RMSE** e **MAE**, mais a cobertura), avaliamos listas de 10 filmes com **Precision@10 e Recall@10**, considerando relevante o que o usuário avaliou com 4 ou 5 no teste. Os resultados foram fatiados por perfil (usuários com menos de 30 avaliações até os com 120 ou mais) e o suporte mínimo foi variado (2 a 40 filmes em comum) para ver o que acontece quando se exige mais evidência.

## O que encontramos

O melhor modelo foi **Pearson com notas centralizadas**, com **RMSE 0,948** e **MAE 0,742**, o que é **7,5% melhor que a média do filme** (1,025), 9,2% melhor que a média do usuário e 15,8% melhor que a média global. Ele ganhou das três referências nas 5 partições.

Três coisas chamaram mais atenção do que o placar:

1. **Centralizar importa mais que escolher a similaridade.** Descontar a média de cada pessoa reduziu o RMSE em cerca de 0,06 para cosseno e Pearson, em todas as partições. Já a diferença entre as três similaridades, uma vez centralizado, é de três milésimos. Sem a centralização, o KNN básico ganha só 1% a 1,5% da média do filme, exceto o MSD, que chega a 5%.
2. **Mais histórico, menos erro.** O RMSE vai de 1,024 (poucas avaliações) a 0,925 (muitas). Isso confirma a parte da H4 que fala em esparsidade, e a curva de *k* mostra que, a partir de uns 40 vizinhos, aumentar o número deixa de ajudar.
3. **Em listas de 10 filmes, o KNN puro perde para os mais populares.** Sem exigir um mínimo de vizinhos, a Precision@10 é 0,001: o topo da lista se enche de filmes obscuros com nota alta vinda de um único vizinho. Exigindo ao menos 20 vizinhos, sobe para 0,093, ainda abaixo dos 0,157 da lista de populares (que ganhou em todas as partições). A diferença diminui com o histórico: o KNN chega a 49% da precisão dos populares para quem tem poucas notas e 67% para quem tem muitas. Um exemplo concreto (usuário 54): *The Shawshank Redemption* previsto em 4,59 com 80 vizinhos, e *The World of Apu* em 4,68 com apenas 5. Notas parecidas, confiança bem diferente.

**Ressalvas.** As 5 partições não são independentes, então os desvios padrão tendem a ser subestimados e não fizemos teste estatístico formal. A avaliação offline favorece títulos populares, o que não significa que pessoas reais prefeririam essa lista. Os dados são de 1997–98. O limiar de 20 vizinhos foi acrescentado *depois* de ver que o de 5 ainda ia mal, portanto é análise de sensibilidade, não ajuste. A conclusão prática é que a nota prevista sozinha não basta: a **confiança** da previsão (quantos vizinhos a sustentam) precisa fazer parte da recomendação.

---

## Reproduzir

Requisitos: Python 3 com numpy, pandas e matplotlib; `pdflatex` com a classe IEEEtran.

```bash
# 1. baixe ml-100k.zip (link acima) e extraia em data/ml-100k/
# 2. experimentos (~2,5 min, semente 42)
python3 run_experiments.py --data data/ml-100k --out results
# 3. figuras, tabelas e números do texto
python3 make_assets.py --results results/results.json --out assets
# 4. artigo (rode três vezes para resolver as referências)
pdflatex article.tex
```

## Estrutura

```
run_experiments.py     KNN usuário-usuário, baselines, seleção de k, métricas, grupos, suporte mínimo
make_assets.py         gera figuras (PDF), tabelas e macros a partir de results/results.json
results/results.json   resultados completos da execução
article.tex, assets/   fonte do artigo (IEEE) com figuras e tabelas geradas
```

## Referências principais

Resnick et al. (1994), GroupLens · Herlocker et al. (1999) · Sarwar et al. (2001) · Desrosiers e Karypis (2011) · Harper e Konstan (2015), MovieLens · Hug (2020), Surprise. A lista completa está no artigo.

## Autores

[Matheus dos Santos] · [Matheus Vasconcelos] · [Pedro Augusto] — [Afya - Maceió], [Ciência da Computação]
