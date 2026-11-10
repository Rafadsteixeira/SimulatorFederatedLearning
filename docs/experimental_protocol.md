# Protocolo experimental

## Unidade de execução

Cada execução recebe um run_id, a configuração efetiva e uma seed. O diretório fl_results_v15/runs/<run_id>/ contém config.json, decisões Agno, eventos dos clientes, métricas das rodadas e um resumo final.

Todo evento de cliente e rodada contém o run_id. Cada atualização convidada termina em um estado terminal persistido.

## Particionamento e reprodutibilidade

O conjunto CIFAR-10 de treino é particionado uma vez por execução com uma distribuição Dirichlet não IID. As partições são disjuntas e cobrem cada índice de treino exatamente uma vez.

A seed controla Python, NumPy, PyTorch, transformações e embaralhamento dos batches. Baselines comparados devem usar a mesma configuração, seed, topologia e partição. Chamadas Agno em modo replay reutilizam a decisão registrada para a mesma fase, rodada e grupo.

## Relógio virtual

Para cada cliente:

- t_down estima o download do modelo por tamanho, largura de banda e latência.
- t_comp estima processamento por número de amostras, épocas, capacidade de CPU e variação configurada.
- t_up estima upload dos parâmetros por tamanho, largura de banda e latência.
- tempo_total = t_down + t_comp + t_up, com o jitter determinístico do perfil.
- arrival_time = round_start_time + tempo_total.
- round_end_time = round_start_time + round_deadline.
- lateness_seconds = max(0, arrival_time - round_end_time).

Os tempos simulam as etapas de comunicação e computação. O sistema não precisa realizar download ou atraso de rede artificial.

## Versão e staleness

Cada atualização armazena base_model_version. Quando considerada para agregação:

staleness_tau = current_model_version - base_model_version

Atraso temporal e staleness por versão são métricas separadas. Atualizações com staleness_tau acima de max_staleness são sempre descartadas.

O peso efetivo baseia-se no número de amostras. Com amortecimento:

effective_weight = n_samples / (1 + staleness_tau) ** gamma

A ação reduce_weight multiplica esse resultado por late_update_weight_factor. O agente escolhe somente a ação; o simulador calcula e aplica os pesos.

## Estados de atualização

| Estado | Significado |
|---|---|
| invited | Cliente escolhido e instrução Flower criada. |
| on_time | Atualização chegou até o deadline e foi aceita. |
| late_pending | Atualização atrasada foi adiada para uma rodada futura. |
| accepted_stale | Atualização pendente chegou e foi aceita com amortecimento de staleness. |
| accepted_reduced | Atualização atrasada foi aceita com peso reduzido. |
| discarded_deadline | Atualização descartada por atraso ou pelo fim do experimento. |
| discarded_staleness | Atualização descartada por exceder max_staleness. |
| failed | Cliente convidado não produziu uma atualização válida. |

Os estados são gravados em client_events.jsonl. Atualizações pendentes são fechadas ao fim do experimento.

## Informatividade e contribuição

O perfil pré-rodada contém distribuição de classes, entropia, cobertura de classes raras e divergência Jensen–Shannon em relação à distribuição global de treino. A cobertura rara usa rare_class_threshold e pode ser zero em um dataset balanceado como CIFAR-10.

A norma da atualização mede magnitude, não contribuição causal. O registro também calcula o alinhamento do delta local com o delta agregado. A política recebe esses sinais históricos junto da taxa de atraso e participação do cliente. Não há afirmação de contribuição leave-one-out.

## Fases da política

### Pré-rodada

Ações possíveis: select e skip. A decisão deve selecionar exatamente n_clients identificadores distintos disponíveis. Clientes com atualização ainda pendente não são convidados novamente.

### Pós-deadline

Para atualizações da rodada que excederam o prazo, a política pode:

- defer: armazenar até sua chegada virtual em rodada futura;
- discard: encerrar sem agregar;
- reduce_weight: agregar nesta rodada com o fator configurado.

Para atualizações adiadas que chegaram, pode:

- accept_stale: agregar com o peso amortecido pelo staleness;
- reduce_weight: aplicar também o fator de atraso;
- discard: encerrar sem agregar.

No último passo de finalização, atualizações ainda pendentes são descartadas para garantir um estado terminal.

## Políticas comparadas

- random: seleção aleatória determinística por seed e rodada;
- fastest: menor tempo total previsto;
- dijkstra: menor custo de caminho até o servidor;
- multi_objective: score fixo de informação, cobertura rara, diversidade e custo temporal;
- Agno shadow: audita a proposta e aplica o baseline;
- Agno active: aplica decisões válidas;
- Agno replay: reutiliza decisões gravadas.

A configuração efetiva, seed, decisões, resultados e métricas de cada política permanecem disponíveis por run_id.
