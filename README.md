# SimulatorFederatedLearning

Simulador experimental de aprendizado federado em uma topologia Edge Computing, com CIFAR-10, Flower e uma política Agno opcional.

## Pergunta de pesquisa

Sob quais condições a participação de clientes lentos, mas com dados informativos, melhora a qualidade do modelo o suficiente para compensar a latência adicional?

**Hipótese:** quando os dados de um cliente acrescentam diversidade ou informação útil, aceitar sua atualização com uma regra explícita de deadline e staleness pode melhorar a validação global frente a políticas que escolhem somente por rapidez ou distância.

## O que significa “lento e informativo”

- **Lento:** o tempo previsto inclui download do modelo, processamento local, upload, latência acumulada e largura de banda do caminho.
- **Informativo:** o perfil verificável contém distribuição de classes, entropia, cobertura de classes raras e divergência Jensen–Shannon em relação ao conjunto de treino.
- **Atraso temporal:** segundos entre a chegada prevista e o fim virtual da rodada.
- **Staleness:** diferença entre a versão atual do modelo e a versão usada para treinar a atualização. São valores registrados separadamente.

O relógio é virtual. Nenhum atraso de rede real é necessário para comparar os cenários. Uma atualização aceita com staleness usa o peso calculado pelo simulador; a política não escolhe valores numéricos.

## Arquitetura experimental

A aplicação cria uma partição não IID, disjunta e reproduzível do conjunto de treino. O Flower executa o treinamento com ClientApp, ServerApp e a Strategy TopologyAwareFedAvg. A Strategy seleciona clientes, estima tempos, aplica deadlines, trata atualizações pendentes, agrega pesos e avalia o modelo global uma vez por rodada no conjunto central de teste.

O modo de adiamento usa a mesma Strategy Flower. Não há um loop alternativo que chame o método fit dos clientes diretamente.

### Políticas comparadas

- seleção aleatória;
- menor tempo total previsto;
- menor custo de caminho por Dijkstra;
- score multiobjetivo fixo de informação, cobertura rara, diversidade e tempo;
- Agno em shadow, active ou replay.

Use a mesma configuração, partição, topologia e seed ao comparar políticas. O modo shadow registra a proposta do LLM e aplica o baseline. O modo active aplica somente ações estruturadas e validadas. O modo replay reutiliza decisões registradas em JSONL.

### Política Agno

O contexto pré-rodada inclui infraestrutura, perfil dos dados, previsão de duração, deadline e histórico do cliente. O contexto pós-deadline inclui chegada, atraso, versão-base, staleness, métricas da atualização e sinais anteriores de contribuição.

As ações são limitadas por fase:

- Pré-rodada: select ou skip.
- Atualização que ainda não chegou: defer, discard ou reduce_weight.
- Atualização pendente que chegou: accept_stale, reduce_weight ou discard.
- Staleness acima do limite e encerramento do experimento: descarte obrigatório.

Cada chamada registra contexto, resposta, ação aplicada, fallback, seed, run_id, duração e hash do evento. O histórico posterior contém taxa de atraso, participação, norma da atualização e alinhamento com a agregação; esses sinais não são tratados isoladamente como contribuição causal.

## Requisitos e instalação

O projeto requer Python 3.12 a 3.14 e usa uv para resolver e instalar as dependências declaradas.

~~~powershell
uv sync
~~~

Para executar a interface:

~~~powershell
uv run python SimulatorFederatedLearning.py
~~~

Para rodar os testes:

~~~powershell
uv run pytest
~~~

Dependências não são instaladas automaticamente durante a inicialização. O CIFAR-10 é baixado quando o conjunto é carregado pela aplicação. Para usar Agno com Groq, copie .env.example para .env e preencha GROQ_API_KEY localmente. O modo baseline não exige chamada ao LLM.

## Configuração

Use config_exemplo.json como base. Os principais campos são:

| Campo | Valores ou exemplo | Uso |
|---|---|---|
| n_clients | 5 | Clientes convidados por rodada, limitado ao pool disponível |
| n_rounds | 5 | Número de rodadas Flower |
| seed | 42 | Seed para partição, seleção, Python, NumPy e PyTorch |
| selection_mode | random, fastest, dijkstra, multi_objective | Baseline pré-rodada |
| deadline_mode | fixed, percentile | Regra para escolher o deadline virtual |
| round_deadline | 12.0 | Duração no modo fixed |
| deadline_percentile | 80.0 | Percentil dos tempos convidados |
| late_update_policy | defer, discard, reduced_weight | Baseline para novas atualizações atrasadas |
| late_update_weight_factor | 0.5 | Fator máximo aplicado à ação reduce_weight |
| max_staleness | 2 | Limite de versões para aceitar uma atualização pendente |
| agent_policy_mode | baseline, shadow, active, replay | Política Agno e modo de auditoria |
| agent_replay_file | vazio ou caminho JSONL | Decisões de origem para replay |

No modo replay, informe o JSONL de uma execução anterior em agent_replay_file. As decisões são associadas pela fase, rodada e grupo de decisão; qualquer decisão incompatível usa o fallback e registra o erro.

## Artefatos de cada execução

Cada execução cria uma pasta fl_results_v15/runs/<run_id>/ com:

- config.json: configuração efetiva e seed;
- decisions.jsonl: propostas, decisões aplicadas e auditoria Agno;
- client_events.jsonl: convite e estado final de cada atualização;
- rounds.jsonl: métricas e avaliação centralizada por rodada;
- summary.json: estado e resumo final da execução.

O gráfico da interface continua salvo em fl_results_v15/fl_cifar10_metrics.png. Os artefatos gerados e o CIFAR-10 não devem ser versionados.

## Estrutura

- SimulatorFederatedLearning.py: interface Tkinter e integração com Flower.
- experiment_core.py: particionamento, métricas, seleção, deadlines e agregação testáveis sem GUI.
- agno_policy.py: schema, validação, modos baseline/shadow/active/replay e auditoria.
- run_artifacts.py: persistência por execução.
- docs/experimental_protocol.md: protocolo e definições das métricas.
- tests/: testes do núcleo experimental, auditoria, configuração e avaliação.

O projeto não usa uma camada MVC separada. A interface chama a execução experimental, enquanto as regras puras de seleção, deadline, particionamento e agregação ficam em módulos testáveis.

## Avaliação

A validação é centralizada: o modelo global é avaliado uma vez por rodada no conjunto de teste CIFAR-10. Os clientes treinam em partições distintas e não carregam o conjunto global de teste.

## Licença

Projeto acadêmico — Programa de Pós-Graduação em Ciência da Computação.
© 2026 Rafael de Souza Teixeira / Orientador: Dr. Paulo Silas Servero. Todos os direitos reservados.
