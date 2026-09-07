# Digital Twin FL â€” CIFAR-10 com Dijkstra

> **SimulaÃ§Ã£o de Aprendizado Federado em Digital Twin**  
> Topologia hexagonal 5Ã—5 Â· Roteamento Dijkstra Â· Dataset CIFAR-10  
> Arquitetura **MVC orientada a objetos** em Python

---

**Autores:** Rafael de Souza Teixeira (Mestrando) / Orientador: Dr. Paulo Silas Servero  
**VersÃ£o:** 3.0.0 Â· **Linguagem:** Python 3.10+  
**Origem:** RefatoraÃ§Ã£o de `migrationf2.2.py` para padrÃ£o MVC

---

## Ãndice

- [VisÃ£o Geral](#visÃ£o-geral)
- [PrÃ©-requisitos](#prÃ©-requisitos)
- [InstalaÃ§Ã£o](#instalaÃ§Ã£o)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquivos de ConfiguraÃ§Ã£o](#arquivos-de-configuraÃ§Ã£o)
- [Interface GrÃ¡fica](#interface-grÃ¡fica)
- [Arquitetura MVC](#arquitetura-mvc)
- [ConfiguraÃ§Ã£o do GitHub](#configuraÃ§Ã£o-do-github)
- [Contribuindo](#contribuindo)
- [LicenÃ§a](#licenÃ§a)

---

## VisÃ£o Geral

Este projeto implementa um **Digital Twin** de uma rede mÃ³vel com aprendizado federado (**Federated Learning**) usando o framework [Flower (flwr)](https://flower.ai/). A rede Ã© modelada como uma grade hexagonal 5Ã—5 de estaÃ§Ãµes base, com um servidor central identificado pela maior *betweenness centrality* e clientes selecionados por distÃ¢ncia Dijkstra.

### Funcionalidades principais

| Funcionalidade | DescriÃ§Ã£o |
|---|---|
| ðŸ—ºï¸ Mapa de rede interativo | VisualizaÃ§Ã£o da topologia hexagonal com thumbnails CIFAR-10 |
| ðŸ¤ Treinamento FL (FedAvg) | Rounds federados com agregaÃ§Ã£o ponderada por amostras |
| ðŸ“Š MÃ©tricas de convergÃªncia | GrÃ¡ficos de Loss e AcurÃ¡cia por round (val e treino) |
| âš™ï¸ Config por JSON | ParÃ¢metros carregÃ¡veis de arquivo local ou URL |
| âš¡ Alta-latÃªncia simulada | SeleÃ§Ã£o periÃ³dica de clientes distantes para testes |
| ðŸŽ¯ DistribuiÃ§Ã£o nÃ£o-IID | PartiÃ§Ã£o CIFAR-10 com dominÃ¢ncia de classe por cliente |

---

## PrÃ©-requisitos

- **Python 3.10** ou superior
- **pip** atualizado
- **Git** instalado
- Sistema Operacional: Windows, Linux ou macOS
- (Opcional) GPU CUDA para treinamento mais rÃ¡pido

> **AtenÃ§Ã£o:** O projeto instala as dependÃªncias automaticamente ao iniciar (`main.py`), mas vocÃª pode instalÃ¡-las manualmente com o `requirements.txt`.

---

## InstalaÃ§Ã£o

### 1. Clone o repositÃ³rio

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

### 3. Instale as dependÃªncias

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

O programa irÃ¡:
1. Instalar automaticamente quaisquer dependÃªncias faltantes.
2. Construir a topologia hexagonal e calcular o servidor central (Dijkstra).
3. Abrir a interface grÃ¡fica.

### Fluxo recomendado na interface

```
1. [âš™ Carregar Config JSON]  â†’  Escolha um arquivo de config ou use os padrÃµes
2. [Carregar CIFAR-10]        â†’  Baixa o dataset (~170 MB) e carrega thumbnails
3. [Treinar FL (Dijkstra)]    â†’  Executa os rounds de treinamento federado
4. [Ver MÃ©tricas FL]          â†’  Exibe e salva os grÃ¡ficos de convergÃªncia
```

---

## Estrutura do Projeto

```
migration_fdl/
â”‚
â”œâ”€â”€ main.py                        â† Ponto de entrada: python main.py
â”œâ”€â”€ manifest.json                  â† Metadados completos do projeto
â”œâ”€â”€ requirements.txt               â† DependÃªncias pip
â”‚
â”œâ”€â”€ config/
â”‚   â”œâ”€â”€ example_config.json        â† Exemplo de configuraÃ§Ã£o de experimento
â”‚   â””â”€â”€ schema.json                â† Schema JSON para validaÃ§Ã£o
â”‚
â”œâ”€â”€ models/                        â—„ CAMADA MODEL (lÃ³gica de domÃ­nio)
â”‚   â”œâ”€â”€ network/
â”‚   â”‚   â”œâ”€â”€ network_switch.py      â† Entidade NetworkSwitch
â”‚   â”‚   â”œâ”€â”€ base_station.py        â† Entidade BaseStation
â”‚   â”‚   â”œâ”€â”€ edge_server.py         â† Entidade EdgeServer
â”‚   â”‚   â”œâ”€â”€ user.py                â† Entidade User
â”‚   â”‚   â”œâ”€â”€ topology.py            â† Grafo nx.Graph com cÃ¡lculo de delay
â”‚   â”‚   â””â”€â”€ topology_builder.py    â† FÃ¡brica: grade hexagonal + Dijkstra
â”‚   â””â”€â”€ federated/
â”‚       â”œâ”€â”€ experiment_config.py   â† ConfiguraÃ§Ã£o carregÃ¡vel por JSON
â”‚       â”œâ”€â”€ fl_history.py          â† HistÃ³rico de mÃ©tricas FL
â”‚       â”œâ”€â”€ cifar_cnn.py           â† CNN PyTorch para CIFAR-10
â”‚       â”œâ”€â”€ fl_client.py           â† Cliente Flower (NumPyClient)
â”‚       â”œâ”€â”€ fl_aggregator.py       â† FedAvg e mÃ©dia ponderada
â”‚       â””â”€â”€ dataset_manager.py     â† Download, partiÃ§Ã£o e thumbnails CIFAR-10
â”‚
â”œâ”€â”€ controllers/                   â—„ CAMADA CONTROLLER (orquestraÃ§Ã£o)
â”‚   â”œâ”€â”€ app_controller.py          â† Orquestrador central da aplicaÃ§Ã£o
â”‚   â”œâ”€â”€ fl_controller.py           â† Rounds FL em thread separada
â”‚   â””â”€â”€ config_controller.py       â† Carga e validaÃ§Ã£o de JSON
â”‚
â”œâ”€â”€ views/                         â—„ CAMADA VIEW (interface Tkinter)
â”‚   â”œâ”€â”€ main_window.py             â† Janela principal e layout
â”‚   â”œâ”€â”€ control_bar.py             â† Barra de botÃµes superior
â”‚   â”œâ”€â”€ network_map_view.py        â† Mapa matplotlib da rede
â”‚   â”œâ”€â”€ rounds_panel.py            â† Lista e detalhes de rounds
â”‚   â”œâ”€â”€ metrics_window.py          â† GrÃ¡ficos de convergÃªncia
â”‚   â””â”€â”€ config_dialog.py           â† DiÃ¡logo de carregamento de JSON
â”‚
â””â”€â”€ utils/
    â”œâ”€â”€ installer.py               â† Auto-instalaÃ§Ã£o de dependÃªncias
    â”œâ”€â”€ power_model.py             â† Modelo de energia (placeholder)
    â””â”€â”€ component_manager.py       â† ExportaÃ§Ã£o do cenÃ¡rio como JSON
```

---

## Arquivos de ConfiguraÃ§Ã£o

O experimento Ã© controlado por um arquivo **JSON**. Use `config/example_config.json` como base.

### Campos disponÃ­veis

| Campo | Tipo | PadrÃ£o | DescriÃ§Ã£o |
|---|---|---|---|
| `n_clients` | `int` | `5` | NÃºmero de clientes FL por round |
| `n_rounds` | `int` | `5` | Total de rounds federados |
| `delay_mode` | `"random"` \| `"fixed"` | `"random"` | Modo de cÃ¡lculo de latÃªncias |
| `fixed_weights` | `dict` | `{}` | LatÃªncias fixas por `bs_id` (sÃ³ no modo `fixed`) |
| `high_latency_periodicity` | `int` | `0` | A cada N rounds, inclui clientes HL (`0` = desativado) |
| `high_latency_fraction` | `float` | `0.3` | FraÃ§Ã£o de clientes de alta latÃªncia |
| `seed` | `int` | `42` | Semente para reprodutibilidade |
| `local_epochs` | `int` | `1` | Ã‰pocas de treino local por round |
| `batch_size` | `int` | `32` | Tamanho do batch no DataLoader |

### Exemplo de configuraÃ§Ã£o

```json
{
  "n_clients": 5,
  "n_rounds": 6,
  "delay_mode": "fixed",
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

## Interface GrÃ¡fica

| BotÃ£o | AÃ§Ã£o |
|---|---|
| `Carregar CIFAR-10` | Baixa o dataset e carrega thumbnails no mapa |
| `Treinar FL (Dijkstra)` | Executa os rounds federados em background |
| `Ver MÃ©tricas FL` | Abre janela com 4 grÃ¡ficos + salva PNG em `fl_results_v15/` |
| `âš™ Carregar Config JSON` | Abre diÃ¡logo para selecionar JSON (arquivo ou URL) |

### Legenda do mapa

| SÃ­mbolo | Significado |
|---|---|
| â­ (amarelo) | Servidor central (maior betweenness centrality) |
| ðŸ”µ Azul escuro | Cliente prÃ³ximo ao servidor |
| ðŸŸ£ Roxo | Cliente distante do servidor |
| â” Verde | Link rÃ¡pido (baixo custo delay/bandwidth) |
| â” Laranja | Link mÃ©dio |
| â” Vermelho | Link lento (alto custo) |

---

## Arquitetura MVC

```
      â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”     callbacks     â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
      â”‚    VIEW      â”‚ â—„â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ â”‚   CONTROLLER     â”‚
      â”‚  (Tkinter)   â”‚ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º â”‚ (AppController   â”‚
      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    eventos UI     â”‚  FLController    â”‚
                                        â”‚  ConfigController)â”‚
                                        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                â”‚
                                                â”‚ chama
                                                Ã¢â€“Â¼
                                        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                        â”‚      MODEL       â”‚
                                        â”‚ (TopologyBuilder â”‚
                                        â”‚  DatasetManager  â”‚
                                        â”‚  CifarCNN        â”‚
                                        â”‚  FedAvg / etc.)  â”‚
                                        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

- **View** nÃ£o conhece os Models â€” recebe dados apenas via callbacks.
- **Controller** orquestra, executa threads e notifica a View.
- **Model** Ã© completamente independente de Tkinter (testÃ¡vel isoladamente).

---

## ConfiguraÃ§Ã£o do GitHub

### 1. Inicialize o repositÃ³rio localmente

```bash
cd migration_fdl
git init
git add .
git commit -m "feat: estrutura MVC inicial â€” Digital Twin FL CIFAR-10"
```

### 2. Crie o repositÃ³rio no GitHub

Acesse [github.com/new](https://github.com/new), defina o nome e crie o repositÃ³rio **sem** inicializar com README (jÃ¡ temos o nosso).

### 3. Conecte e envie

```bash
git remote add origin https://github.com/<seu-usuario>/<nome-do-repositorio>.git
git branch -M main
git push -u origin main
```

### 4. `.gitignore` recomendado

Crie um arquivo `.gitignore` na raiz do projeto com o seguinte conteÃºdo:

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

# Arquivos temporÃ¡rios do Windows
Thumbs.db
desktop.ini
tempCodeRunnerFile.py

# Logs
logs/
*.log
```

> **Importante:** O dataset CIFAR-10 (~170 MB) Ã© baixado automaticamente pelo programa e **nÃ£o deve** ser commitado no repositÃ³rio.

### 5. AtualizaÃ§Ãµes futuras

```bash
# Verificar alteraÃ§Ãµes
git status

# Adicionar e commitar mudanÃ§as
git add .
git commit -m "feat: descriÃ§Ã£o da mudanÃ§a"

# Enviar para o GitHub
git push
```

### 6. Usando GitHub Raw para carregar configs

ApÃ³s fazer push dos seus arquivos JSON de config, vocÃª pode carregÃ¡-los diretamente pela interface usando a URL Raw do GitHub:

```
https://raw.githubusercontent.com/<usuario>/<repositorio>/main/config/example_config.json
```

---

## Contribuindo

1. FaÃ§a um **fork** do repositÃ³rio.
2. Crie uma branch para sua feature: `git checkout -b feat/minha-feature`
3. FaÃ§a suas alteraÃ§Ãµes seguindo a arquitetura MVC.
4. Commit com mensagens descritivas: `git commit -m "feat: adiciona novo modo de agregaÃ§Ã£o"`
5. Envie um **Pull Request**.

---

## DependÃªncias

| Pacote | VersÃ£o mÃ­nima | Uso |
|---|---|---|
| `matplotlib` | â‰¥ 3.7 | VisualizaÃ§Ã£o do mapa e grÃ¡ficos |
| `numpy` | â‰¥ 1.24 | OperaÃ§Ãµes numÃ©ricas e pesos |
| `networkx` | â‰¥ 3.1 | Topologia de rede e Dijkstra |
| `torch` | â‰¥ 2.0 | Rede neural CifarCNN |
| `torchvision` | â‰¥ 0.15 | Dataset CIFAR-10 |
| `flwr` | â‰¥ 1.5 | Framework de Aprendizado Federado |
| `Pillow` | â‰¥ 10.0 | ManipulaÃ§Ã£o de imagens |

---

## LicenÃ§a

Projeto acadÃªmico â€” Programa de PÃ³s-GraduaÃ§Ã£o em CiÃªncia da ComputaÃ§Ã£o.  
Â© 2026 Rafael / Orientador: Dr. Paulo Silas Servero. Todos os direitos reservados.



