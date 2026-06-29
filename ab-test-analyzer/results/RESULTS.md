# A/B Test Analyzer, resultados

Rodei um teste A/B com dois grupos de 6.000 usuários cada e olhei tanto pela ótica frequentista quanto pela bayesiana. Os números de conversão por grupo ficaram assim:

| Grupo | Usuários | Conversões | Taxa |
|---|---|---|---|
| Controle (A)   | 6,000 | 593 | 9.883% |
| Tratamento (B) | 6,000 | 735 | 12.250% |

## Conversão

- Lift relativo de **+23.9%** (absoluto de +2.367%).
- Intervalo de confiança de 95% para a diferença: [+1.245%, +3.488%].
- Teste z de duas proporções: z = 4.13, p = 0.0000.
- Conferência cruzada com qui-quadrado: chi2 = 17.07, p = 0.0000.
- Significativo a alpha=0.05? Sim.

## Receita por convertido (métrica secundária)

- Média do controle R$ 43.42 contra R$ 45.96 do tratamento.
- Teste t de Welch: t = 3.86, p = 0.0001.

## Leitura bayesiana (Beta-Binomial, prior plano)

- P(tratamento > controle) = 100.0%. Esse é o número útil para decisão que um p-valor não te dá de forma direta.
- Lift absoluto esperado de +2.367%, com intervalo de credibilidade de 95% em [+1.242%, +3.491%].
- Lendo as duas escolas juntas: o teste frequentista diz "isto não é ruído"; a posterior bayesiana diz "e aqui está o quão confiantes estamos de que B vence, e por quanto".

## Poder e tamanho de amostra (desenhar o teste, não só ler)

- Para detectar um lift relativo de +10% com 80% de poder (alpha=0.05), você precisaria de cerca de 14.946 usuários por braço.
- Este teste (6.000 por braço) tem poder para detectar até um lift relativo de +16.0% com 80% de poder. Qualquer coisa menor do que isso ele provavelmente deixaria passar.
- Poder pós-hoc para o lift observado de +23.9%: 99%.
- Resumo da ópera: o efeito observado foi grande o bastante para ser detectado, mas um lift real abaixo de cerca de +16.0% teria deixado este teste subdimensionado. Calibre o próximo experimento para o menor lift que valha a pena colocar em produção.

## Veredito

Sobe o B. O ganho é real e dificilmente é ruído.
