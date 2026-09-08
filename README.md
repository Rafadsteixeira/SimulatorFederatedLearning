# Digital Twin FL — CIFAR-10 com Dijkstra

> **Simulação de Aprendizado Federado em Digital Twin**  
> Topologia hexagonal 5×5 · Roteamento Dijkstra · Dataset CIFAR-10  
> Arquitetura **MVC orientada a objetos** em Python

---

**Autores:** Rafael de Souza Teixeira (Mestrando) / Orientador: Dr. Paulo Silas Servero  
**Versão:** 3.0.0 · **Linguagem:** Python 3.10+  
**Origem:** Refatoração de `migrationf2.2.py` para padrão MVC

---

## Índice

- [Visão Geral](#visão-geral)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquivos de Configuração](#arquivos-de-configuração)
- [Interface Gráfica](#interface-gráfica)
- [Arquitetura MVC](#arquitetura-mvc)
- [Configuração do GitHub](#configuração-do-github)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

---

## Visão Geral

Este projeto implementa um **Digital Twin** de uma rede móvel com aprendizado federado (**Federated Learning**) usando o framework [Flower (flwr)](https://flower.ai/). A rede é modelada como uma grade hexagonal 5×5 de estações base, com um servidor central identificado pela maior *betweenness centrality* e clientes selecionados por distância Dijkstra.

### Funcionalidades principais

| Funcionalidade | Descrição |
|---|---|
| 🗺️ Mapa de rede interativo | Visualização da topologia hexagonal com thumbnails CIFAR-10 |
| 🤝 Treinamento FL (FedAvg) | Rounds federados com agregação ponderada por amostras |
| 📊 Métricas de convergência | Gráficos de Loss e Acurácia por round (val e treino) |
| ⚙️ Config por JSON | Parâmetros carregáveis de arquivo local ou URL |
| ⚡ Alta-latência simulada | Seleção periódica de clientes distantes para testes |
| 🎯 Distribuição não-IID | Partição CIFAR-10 com dominância de classe por cliente |

---

## Pré-requisitos

- **Python 3.10** ou superior
- **pip** atualizado
- **Git** instalado
- Sistema Operacional: Windows, Linux ou macOS
- (Opcional) GPU CUDA para treinamento mais rápido

> **Atenção:** O projeto instala as dependências automaticamente ao iniciar (`main.py`), mas você pode instalá-las manualmente com o `requirements.txt`.

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
```

### 2. (Recomendado) Crie um ambiente virtual

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
pip install -r requirements.txt
```

> **GPU (opcional):** Para usar CUDA, instale o PyTorch com suporte CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Como Executar

```bash
python main.py
```

O programa irá:
1. Instalar automaticamente quaisquer dependências faltantes.
2. Construir a topologia hexagonal e calcular o servidor central (Dijkstra).
3. Abrir a interface gráfica.

### Fluxo recomendado na interface

```
1. [⚙ Carregar Config JSON]  →  Escolha um arquivo de config ou use os padrões
2. [Carregar CIFAR-10]        →  Baixa o dataset (~170 MB) e carrega thumbnails
3. [Treinar FL (Dijkstra)]    →  Executa os rounds de treinamento federado
4. [Ver Métricas FL]          →  Exibe e salva os gráficos de convergência
```

---

## Estrutura do Projeto

```
migration_fdl/
│
├── main.py                        ← Ponto de entrada: python main.py
├── manifest.json                  ← Metadados completos do projeto
├── requirements.txt               ← Dependências pip
│
├── config/
│   ├── example_config.json        ← Exemplo de configuração de experimento
│   └── schema.json                ← Schema JSON para validação
│
├── models/                        ◄ CAMADA MODEL (lógica de domínio)
│   ├── network/
│   │   ├── network_switch.py      ← Entidade NetworkSwitch
│   │   ├── base_station.py        ← Entidade BaseStation
│   │   ├── edge_server.py         ← Entidade EdgeServer
│   │   ├── user.py                ← Entidade User
│   │   ├── topology.py            ← Grafo nx.Graph com cálculo de delay
│   │   └── topology_builder.py    ← Fábrica: grade hexagonal + Dijkstra
│   └── federated/
│       ├── experiment_config.py   ← Configuração carregável por JSON
│       ├── fl_history.py          ← Histórico de métricas FL
│       ├── cifar_cnn.py           ← CNN PyTorch para CIFAR-10
│       ├── fl_client.py           ← Cliente Flower (NumPyClient)
│       ├── fl_aggregator.py       ← FedAvg e média ponderada
│       └── dataset_manager.py     ← Download, partição e thumbnails CIFAR-10
│
├── controllers/                   ◄ CAMADA CONTROLLER (orquestração)
│   ├── app_controller.py          ← Orquestrador central da aplicação
│   ├── fl_controller.py           ← Rounds FL em thread separada
│   └── config_controller.py       ← Carga e validação de JSON
│
├── views/                         ◄ CAMADA VIEW (interface Tkinter)
│   ├── main_window.py             ← Janela principal e layout
│   ├── control_bar.py             ← Barra de botões superior
│   ├── network_map_view.py        ← Mapa matplotlib da rede
│   ├── rounds_panel.py            ← Lista e detalhes de rounds
│   ├── metrics_window.py          ← Gráficos de convergência
│   └── config_dialog.py           ← Diálogo de carregamento de JSON
│
└── utils/
    ├── installer.py               ← Auto-instalação de dependências
    ├── power_model.py             ← Modelo de energia (placeholder)
    └── component_manager.py       ← Exportação do cenário como JSON
```

---

## Arquivos de Configuração

O experimento é controlado por um arquivo **JSON**. Use `config/example_config.json` como base.

### Campos disponíveis

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `n_clients` | `int` | `5` | Número de clientes FL por round |
| `n_rounds` | `int` | `5` | Total de rounds federados |
| `weight_mode` | `"random"` \| `"fixed"` | `"random"` | Modo de cálculo de latências |
| `fixed_weights` | `dict` | `{}` | Latências fixas por `bs_id` (só no modo `fixed`) |
| `high_latency_periodicity` | `int` | `0` | A cada N rounds, inclui clientes HL (`0` = desativado) |
| `high_latency_fraction` | `float` | `0.3` | Fração de clientes de alta latência |
| `seed` | `int` | `42` | Semente para reprodutibilidade |
| `local_epochs` | `int` | `1` | Épocas de treino local por round |
| `batch_size` | `int` | `32` | Tamanho do batch no DataLoader |

### Exemplo de configuração

```json
{
  "n_clients": 5,
  "n_rounds": 6,
  "weight_mode": "fixed",
  "fixed_weights": {
    "1": 2.5,
    "2": 5.1,
    "3": 1.8,
    "4": 8.3,
    "5": 3.2
  },
  "high_latency_periodicity": 3,
  "high_latency_fraction": 0.3,
  "seed": 42,
  "local_epochs": 1,
  "batch_size": 32
}
```

> O arquivo pode ser carregado pela interface via **arquivo local** ou **URL remota** (ex.: GitHub Raw).

---

## Interface Gráfica

| Botão | Ação |
|---|---|
| `Carregar CIFAR-10` | Baixa o dataset e carrega thumbnails no mapa |
| `Treinar FL (Dijkstra)` | Executa os rounds federados em background |
| `Ver Métricas FL` | Abre janela com 4 gráficos + salva PNG em `fl_results_v15/` |
| `⚙ Carregar Config JSON` | Abre diálogo para selecionar JSON (arquivo ou URL) |

### Legenda do mapa

| Símbolo | Significado |
|---|---|
| ⭐ (amarelo) | Servidor central (maior betweenness centrality) |
| 🔵 Azul escuro | Cliente próximo ao servidor |
| 🟣 Roxo | Cliente distante do servidor |
| ━ Verde | Link rápido (baixo custo delay/bandwidth) |
| ━ Laranja | Link médio |
| ━ Vermelho | Link lento (alto custo) |

---

## Arquitetura MVC

```
      ┌─────────────┐     callbacks     ┌──────────────────┐
      │    VIEW      │ ◄──────────────── │   CONTROLLER     │
      │  (Tkinter)   │ ───────────────► │ (AppController   │
      └─────────────┘    eventos UI     │  FLController    │
                                        │  ConfigController)│
                                        └──────────────────┘
                                                │
                                                │ chama
                                                â–¼
                                        ┌──────────────────┐
                                        │      MODEL       │
                                        │ (TopologyBuilder │
                                        │  DatasetManager  │
                                        │  CifarCNN        │
                                        │  FedAvg / etc.)  │
                                        └──────────────────┘
```

- **View** não conhece os Models — recebe dados apenas via callbacks.
- **Controller** orquestra, executa threads e notifica a View.
- **Model** é completamente independente de Tkinter (testável isoladamente).

---

## Configuração do GitHub

### 1. Inicialize o repositório localmente

```bash
cd migration_fdl
git init
git add .
git commit -m "feat: estrutura MVC inicial — Digital Twin FL CIFAR-10"
```

### 2. Crie o repositório no GitHub

Acesse [github.com/new](https://github.com/new), defina o nome e crie o repositório **sem** inicializar com README (já temos o nosso).

### 3. Conecte e envie

```bash
git remote add origin https://github.com/<seu-usuario>/<nome-do-repositorio>.git
git branch -M main
git push -u origin main
```

### 4. `.gitignore` recomendado

Crie um arquivo `.gitignore` na raiz do projeto com o seguinte conteúdo:

```gitignore
# Ambiente virtual
.venv/
venv/
env/

# Cache Python
__pycache__/
*.py[cod]
*.pyo

# Dataset CIFAR-10 (muito grande para o Git)
data_cifar10/

# Resultados gerados
fl_results_v15/
datasets/
*.png
*.csv

# Arquivos do VS Code / IDE
.vscode/
.idea/

# Arquivos temporários do Windows
Thumbs.db
desktop.ini
tempCodeRunnerFile.py

# Logs
logs/
*.log
```

> **Importante:** O dataset CIFAR-10 (~170 MB) é baixado automaticamente pelo programa e **não deve** ser commitado no repositório.

### 5. Atualizações futuras

```bash
# Verificar alterações
git status

# Adicionar e commitar mudanças
git add .
git commit -m "feat: descrição da mudança"

# Enviar para o GitHub
git push
```

### 6. Usando GitHub Raw para carregar configs

Após fazer push dos seus arquivos JSON de config, você pode carregá-los diretamente pela interface usando a URL Raw do GitHub:

```
https://raw.githubusercontent.com/<usuario>/<repositorio>/main/config/example_config.json
```

---

## Contribuindo

1. Faça um **fork** do repositório.
2. Crie uma branch para sua feature: `git checkout -b feat/minha-feature`
3. Faça suas alterações seguindo a arquitetura MVC.
4. Commit com mensagens descritivas: `git commit -m "feat: adiciona novo modo de agregação"`
5. Envie um **Pull Request**.

---

## Dependências

| Pacote | Versão mínima | Uso |
|---|---|---|
| `matplotlib` | ≥ 3.7 | Visualização do mapa e gráficos |
| `numpy` | ≥ 1.24 | Operações numéricas e pesos |
| `networkx` | ≥ 3.1 | Topologia de rede e Dijkstra |
| `torch` | ≥ 2.0 | Rede neural CifarCNN |
| `torchvision` | ≥ 0.15 | Dataset CIFAR-10 |
| `flwr` | ≥ 1.5 | Framework de Aprendizado Federado |
| `Pillow` | ≥ 10.0 | Manipulação de imagens |

---

## Licença

Projeto acadêmico — Programa de Pós-Graduação em Ciência da Computação.  
© 2026 Rafael / Orientador: Dr. Paulo Silas Servero. Todos os direitos reservados.


