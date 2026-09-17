# Simulator Federated Learning — CIFAR-10

> **Simulação de aprendizado federado em uma topologia Edge Computing**<br>
> Grade hexagonal 5×5 · Roteamento por Dijkstra · Dataset CIFAR-10 · Flower

---

**Autores:** Rafael de Souza Teixeira (Mestrando) / Orientador: Dr. Paulo Silas Servero<br>
**Versão:** 3.0.0 · **Linguagem:** Python 3.10+<br>
**Repositório:** [Rafadsteixeira/SimulatorFederatedLearning](https://github.com/Rafadsteixeira/SimulatorFederatedLearning)<br><br>
**Research Questions** - Sob quais condições a inclusão de clientes lentos, porém com dados informativos, compensa o custo adicional de latência no processo de treinamento distribuído em dispositivos móveis?
---

## Índice

- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Fluxo da Interface](#fluxo-da-interface)
- [Configuração do Experimento](#configuração-do-experimento)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquitetura](#arquitetura)
- [Resultados](#resultados)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## Visão Geral

O projeto simula uma rede de Edge Computing com aprendizado federado sobre o
dataset CIFAR-10. A rede é representada por uma grade hexagonal de estações
base, conectadas por enlaces com latência e largura de banda simuladas.

O treinamento é executado com [Flower](https://flower.ai/), usando clientes
federados que treinam uma CNN localmente e enviam apenas os pesos do modelo ao
servidor. A estratégia `TopologyAwareFedAvg` combina FedAvg com:

- seleção de clientes baseada na topologia e em caminhos mínimos de Dijkstra;
- simulação de tempo de comunicação e computação;
- clientes e dados não-IID;
- rounds com alta latência;
- deadlines, clientes atrasados e staleness;
- métricas de treino e validação por round.

---

## Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| Mapa interativo | Exibe a topologia, estações base, servidor central e enlaces |
| CIFAR-10 | Baixa e carrega automaticamente os conjuntos de treino e teste |
| Treinamento FL | Executa rounds com Flower, ClientApp, ServerApp e FedAvg |
| Seleção de clientes | Usa latência ou otimização multiobjetivo |
| Alta latência | Permite incluir clientes distantes em rounds periódicos |
| Staleness | Trata atualizações atrasadas conforme a política configurada |
| Métricas | Exibe loss e acurácia de treino e validação |
| Configuração JSON | Carrega configurações de arquivo local ou URL |
| Exportação | Salva o cenário da topologia e os gráficos gerados |

---

## Pré-requisitos

- Python 3.10 ou superior;
- `pip`;
- Tkinter, normalmente incluído na instalação do Python;
- conexão com a internet para baixar o CIFAR-10 e dependências ausentes;
- GPU CUDA opcional.

O programa verifica e tenta instalar automaticamente as dependências Python
ausentes ao iniciar. Também é possível instalá-las manualmente:

```bash
pip install matplotlib numpy networkx torch torchvision "flwr[simulation]>=1.6" Pillow
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/Rafadsteixeira/SimulatorFederatedLearning.git
cd SimulatorFederatedLearning
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install matplotlib numpy networkx torch torchvision "flwr[simulation]>=1.6" Pillow
```

Para instalação do PyTorch com CUDA, consulte a
[página oficial do PyTorch](https://pytorch.org/get-started/locally/) e use o
comando correspondente à sua versão do CUDA.

---

## Como Executar

```bash
python SimulatorFederatedLearning.py
```

O programa abrirá a interface gráfica e preparará a topologia da rede. O
dataset CIFAR-10 é baixado somente quando a opção de carregamento é acionada.

---

## Fluxo da Interface

```text
1. Carregar Dataset  → baixa o CIFAR-10 e carrega imagens de exemplo
2. Carregar Config   → opcionalmente seleciona um JSON local ou uma URL
3. Treinar Modelo    → executa os rounds de aprendizado federado
4. Ver Métricas FL   → exibe os gráficos e o resumo do treinamento
```

Durante o treinamento, a interface mostra o status atual, a lista de rounds e
detalhes de clientes, incluindo tempo estimado, deadline, loss e norma da
atualização.

---

## Configuração do Experimento

O arquivo [config_exemplo.json](config_exemplo.json) pode ser usado como base.
Ele pode ser carregado pela interface a partir de um arquivo local ou de uma
URL acessível pela aplicação.

### Campos disponíveis

| Campo | Tipo | Padrão | Descrição |
|---|---:|---:|---|
| `n_clients` | `int` | `5` | Clientes selecionados por round |
| `n_rounds` | `int` | `5` | Número de rounds federados |
| `weight_mode` | `string` | `"random"` | Modo de pesos/latências |
| `fixed_weights` | `object` | `{}` | Pesos fixos por estação quando aplicável |
| `high_latency_periodicity` | `int` | `0` | Frequência de rounds de alta latência |
| `high_latency_fraction` | `float` | `0.3` | Fração de clientes de alta latência |
| `seed` | `int` | `42` | Semente de reprodutibilidade |
| `local_epochs` | `int` | `1` | Épocas de treino local |
| `batch_size` | `int` | `32` | Tamanho do batch |
| `deadline_mode` | `string` | `"percentile"` | `"fixed"` ou `"percentile"` |
| `round_deadline` | `float` | `12.0` | Deadline em segundos no modo fixo |
| `deadline_percentile` | `float` | `80.0` | Percentil usado para definir o deadline |
| `comp_time_per_sample` | `float` | `0.005` | Tempo computacional simulado por amostra |
| `enable_staleness` | `bool` | `true` | Ativa a penalização de staleness |
| `staleness_gamma` | `float` | `0.5` | Expoente da penalização de staleness |
| `late_update_policy` | `string` | `"reduced_weight"` | `"discard"`, `"defer"` ou `"reduced_weight"` |
| `max_staleness` | `int` | `2` | Staleness máximo aceito |
| `selection_mode` | `string` | `"multi_objective"` | `"latency"` ou `"multi_objective"` |
| `selection_info_weight` | `float` | `0.45` | Peso da informação na seleção |
| `selection_rare_weight` | `float` | `0.25` | Peso de classes raras |
| `selection_diversity_weight` | `float` | `0.15` | Peso da diversidade |
| `selection_time_weight` | `float` | `0.15` | Peso do tempo |
| `rare_class_threshold` | `float` | `0.08` | Limite para identificar classes raras |
| `time_jitter_fraction` | `float` | `0.0` | Variação aleatória do tempo simulado |

Exemplo mínimo:

```json
{
  "n_clients": 5,
  "n_rounds": 5,
  "weight_mode": "random",
  "seed": 42,
  "local_epochs": 1,
  "batch_size": 32,
  "deadline_mode": "percentile",
  "selection_mode": "multi_objective"
}
```

---

## Estrutura do Projeto

```text
SimulatorFederatedLearning/
├── SimulatorFederatedLearning.py  # Aplicação, modelo, topologia e GUI
├── config_exemplo.json             # Exemplo de configuração
├── .gitignore                      # Datasets e artefatos gerados
├── data_cifar10/                   # CIFAR-10 baixado localmente (não versionado)
├── datasets/                       # Cenários exportados (gerado em execução)
└── fl_results_v15/                 # Gráficos e resultados (gerado em execução)
```

O projeto atual concentra a implementação em um único arquivo Python. Os
diretórios de dados e resultados são criados automaticamente quando
necessários.

---

## Arquitetura

```text
┌──────────────────────┐
│ Interface Tkinter    │
│ mapa, controles, logs│
└──────────┬───────────┘
           │ callbacks
┌──────────▼───────────┐
│ Treinamento FL       │
│ Flower + FedAvg      │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Topologia e modelo   │
│ NetworkX + PyTorch   │
└──────────────────────┘
```

O `CifarFederatedClient` treina uma partição local do CIFAR-10. A estratégia
`TopologyAwareFedAvg` seleciona clientes, calcula tempos simulados e agrega
os pesos. A interface é atualizada por callbacks enquanto o treinamento roda
em uma thread separada.

---

## Resultados

Ao abrir a janela de métricas, o gráfico é salvo em:

```text
fl_results_v15/fl_cifar10_metrics.png
```

O cenário da rede pode ser exportado em:

```text
datasets/sample_dataset_v15.json
```

O dataset CIFAR-10 e os resultados gerados não devem ser versionados, pois os
arquivos do dataset excedem o limite de tamanho do GitHub. O `.gitignore`
existente já cobre esses diretórios.

---

## Contribuindo

1. Faça um fork do repositório.
2. Crie uma branch: `git checkout -b feat/minha-feature`.
3. Faça a alteração e teste a execução local.
4. Verifique o estado do Git com `git status`.
5. Crie um commit descritivo.
6. Envie a branch e abra um Pull Request.

Não inclua `data_cifar10/`, `datasets/`, `fl_results_v15/` ou caches Python
nos commits.

---

## Licença

Projeto acadêmico — Programa de Pós-Graduação em Ciência da Computação.
© 2026 Rafael de Souza Teixeira / Orientador: Dr. Paulo Silas Servero. Todos os direitos reservados.
