# Portfólio de ciência de dados

Três estudos em Python com dados simulados e semente fixa. Os resultados
demonstram como avaliar modelos e não estimam desempenho em produção.

## Como rodar

Use Python 3.9 ou superior. Não há dependências externas.

```sh
python portfolio.py voos
python portfolio.py transacoes
python portfolio.py bicicletas
```

## Atrasos em voos

**Pergunta:** quais voos têm maior chance de atrasar mais de 15 minutos?
A regressão logística usa clima, congestionamento e horário da partida
como atributos simulados conhecidos antes do voo. O teste usa os últimos
20% dos registros em ordem temporal.

Dos 100 voos com maior escore, 66 atrasaram. A prevalência no teste foi de
36%, e a precisão média foi 0,582.

## Transações suspeitas

**Pergunta:** quais transações merecem revisão quando a capacidade da equipe é
limitada? O modelo usa valor, localização incomum e frequência recente de
transações. O limiar foi escolhido na validação para sinalizar cerca de 5%
dos casos. O conjunto de teste é posterior e tem 3% de positivos.

O limiar gerou 53 alertas no teste. Houve 9 acertos, com precisão de 17,0%
e recall de 25,0%. O escore serve para priorizar uma revisão manual.

## Bicicletas compartilhadas

**Pergunta:** quantas retiradas ocorrerão na próxima hora? A avaliação
avança hora a hora em quatro semanas de teste e compara duas previsões
sazonais que usam apenas retiradas já observadas.

| Método | Erro absoluto médio |
| --- | ---: |
| Mesmo horário da semana anterior | 5,72 retiradas/hora |
| Média do mesmo horário nas últimas quatro semanas | 4,63 retiradas/hora |

## Limitações

Os geradores têm relações conhecidas entre atributos e desfechos. Uma
avaliação com dados reais exigiria separar períodos completos, verificar
quando cada atributo fica disponível e medir como o desempenho muda com o tempo.
