

## 1) Nome do projeto e identidade visual

### Problema
O projeto aparece com mÃºltiplos nomes diferentes, sem um padrÃ£o Ãºnico:
- `SimulatorFederatedLearning`
- `SimulatorFederatedLearning`
- `SimulatorFederatedLearning`
- `SimulatorFederatedLearning`

Isso dificulta a compreensÃ£o da identidade do software e cria ambiguidades na documentaÃ§Ã£o e na interface.

### SugestÃ£o
Escolher um nome principal e usÃ¡-lo de forma consistente em:
- README
- manifest.json
- tÃ­tulos de janela
- mensagens da interface
- comentÃ¡rios de mÃ³dulo
- nomes de artefatos exportados

### RecomendaÃ§Ã£o
Usar como nome principal:
- `SimulatorFederatedLearning`

Usar como nome tÃ©cnico (se necessÃ¡rio):
- `simulator_federated_learning`

Usar como nome do repositÃ³rio:
- `SimulatorFederatedLearning`

## 2) Nome do repositÃ³rio

### Atual
- `SimulatorFederatedLearning`

### Problema
O nome anterior estava grafado incorretamente; a versÃ£o padronizada passa a ser `SimulatorFederatedLearning`.

### RecomendaÃ§Ã£o
- `SimulatorFederatedLearning`

## 3) Identidade do mÃ³dulo principal

### Atual
- `SimulatorFederatedLearning`

### Problema
Esse nome deve ser usado de forma consistente em toda a documentaÃ§Ã£o, interface e arquivos de configuraÃ§Ã£o, em vez de referÃªncias histÃ³ricas ou legadas.

### RecomendaÃ§Ã£o
Usar apenas:
- `SimulatorFederatedLearning`
- ou `simulator_federated_learning` em nomes tÃ©cnicos

## 4) DocumentaÃ§Ã£o e estrutura do projeto

### Problema
A documentaÃ§Ã£o deve refletir a estrutura atual do projeto sem nomes antigos ou artificiais.

### RecomendaÃ§Ã£o
Padronizar a documentaÃ§Ã£o para refletir o nome oficial do projeto:
- `SimulatorFederatedLearning/`

## 5) Nomes de classes

### `FederatedClient`
- Atual: `ImageMigrationClient`
- Problema: o termo `migration` nÃ£o combina com o domÃ­nio atual de simulaÃ§Ã£o de aprendizado federado em CIFAR-10.
- RecomendaÃ§Ã£o:
  - `FederatedClient`
  - ou `CifarFederatedClient`

### `FLHistory`
- Atual: `FlHistory`
- Problema: o nome mistura convenÃ§Ãµes diferentes. O padrÃ£o do projeto usa `FL` em maiÃºsculas em muitos lugares, mas a classe usa `FlHistory` em camel-case.
- RecomendaÃ§Ã£o:
  - `FLHistory`

### `ComponentManager`
- Atual: `ComponentManager`
- Problema: o nome Ã© genÃ©rico e nÃ£o comunica bem a funÃ§Ã£o real do mÃ³dulo.
- RecomendaÃ§Ã£o:
  - `ScenarioExporter`
  - ou `TopologyExporter`

### `ServerPowerModel`
- Atual: `LinearServerPowerModel`
- Problema: nome aceitÃ¡vel, mas genÃ©rico e pouco usado na lÃ³gica atual.
- RecomendaÃ§Ã£o:
  - `ServerPowerModel`

## 6) Nomes de termos de domÃ­nio e configuraÃ§Ã£o

### `delay_mode`
- Atual: `delay_mode`
- Problema: no cÃ³digo real, esse campo controla latÃªncia/atraso de roteamento, e nÃ£o peso do modelo.
- RecomendaÃ§Ã£o:
  - `delay_mode`
  - ou `routing_delay_mode`

### `fixed_weights`
- Atual: `fixed_weights`
- Problema: o termo `weights` nÃ£o corresponde ao conceito real, que Ã© latÃªncia fixa.
- RecomendaÃ§Ã£o:
  - `fixed_delays`
  - ou `fixed_route_delays`

### `delay_mode == "random"`
- Atual: `"random"`
- RecomendaÃ§Ã£o:
  - `"random_delay"`

### `delay_mode == "fixed"`
- Atual: `"fixed"`
- RecomendaÃ§Ã£o:
  - `"fixed_delay"`

## 7) Conceitos de rede vs clientes

### Problema
O código usa a palavra `base_stations` em vários pontos para se referir a `BaseStation`, enquanto o domínio real é uma estação base e não um cliente usuário.

Exemplos:
- `base_stations`
- `selected_base_stations`
- texto da interface: “Estação base (próxima)” e “Estação base (distante)”

### Recomendação
Usar nomes mais específicos:
- `base_stations`
- `selected_base_stations`
- “Estação base (próxima)” / “Estação base (distante)”

## 8) Interface e textos visuais

### TÃ­tulo da janela principal
- Atual: `SimulatorFederatedLearning â€” CIFAR-10 | 1 Servidor Central + Dijkstra`
- Problema: a identidade do projeto deve ficar explÃ­cita e consistente.
- RecomendaÃ§Ã£o:
  - `SimulatorFederatedLearning â€” CIFAR-10 | Servidor Central + Dijkstra`

### BotÃ£o de configuraÃ§Ã£o
- Atual: `âš™ Carregar Config JSON`
- RecomendaÃ§Ã£o:
  - `âš™ Carregar configuraÃ§Ã£o JSON`

### BotÃ£o de treinamento
- Atual: `Treinar FL (Dijkstra)`
- RecomendaÃ§Ã£o:
  - `Treinar Modelo`

### BotÃ£o de mÃ©tricas
- Atual: `Ver MÃ©tricas FL`
- RecomendaÃ§Ã£o:
  - `Ver mÃ©tricas FL`

### Status da barra
- Atual: `FL: NÃ£o treinado`
- RecomendaÃ§Ã£o:
  - `Status: NÃ£o treinado`

### Mensagem de configuraÃ§Ã£o vazia
- Atual: `Nenhum JSON carregado. Usando valores padrÃ£o.`
- RecomendaÃ§Ã£o:
  - `Nenhum arquivo de configuraÃ§Ã£o carregado. Usando valores padrÃ£o.`

### Mensagem de config carregada
- Atual: `Config carregada`
- RecomendaÃ§Ã£o:
  - `ConfiguraÃ§Ã£o carregada`

### Mensagem de erro de JSON
- Atual: `Erro ao carregar JSON`
- RecomendaÃ§Ã£o:
  - `Erro ao carregar arquivo JSON`

### Mensagem de erro de treinamento
- Atual: `Erro no Treinamento FL`
- RecomendaÃ§Ã£o:
  - `Erro no treinamento FL`

## 9) ComentÃ¡rios e docstrings

### Problema
HÃ¡ vÃ¡rios comentÃ¡rios e docstrings que ainda apresentam nomes legados ou misturam conceitos antigos com o estado atual do projeto.

### Exemplos
- `main.py` descreve a aplicaÃ§Ã£o como `SimulatorFederatedLearning`
- README menciona `SimulatorFederatedLearning` como origem
- vÃ¡rios trechos falam em â€œSimulatorFederatedLearningâ€ como nome da estrutura do projeto

### RecomendaÃ§Ã£o
Usar em docstrings e comentÃ¡rios:
- `SimulatorFederatedLearning`
- `aplicaÃ§Ã£o SimulatorFederatedLearning`
- `arquitetura MVC`
- `simulaÃ§Ã£o de aprendizagem federada`

Evitar:
- nomes legados e variantes histÃ³ricas

## 10) Nomes de diretÃ³rios de saÃ­da

### Atual
- `./fl_results_v15`
- `./data_cifar10`
- `datasets/`

### Problema
HÃ¡ confusÃ£o entre artefatos de resultados, dados e cenÃ¡rio exportado.

### RecomendaÃ§Ã£o
- usar `results/` para mÃ©tricas e grÃ¡ficos
- usar `data/` para datasets
- usar `scenarios/` ou `network_scenarios/` para exportaÃ§Ãµes de topologia

Exemplos:
- `results/fl_metrics`
- `data/cifar10`
- `scenarios/hexagonal_5x5.json`

## 11) Nomes de arquivos gerados

### Atual
- `sample_dataset_v15.json`
- `fl_cifar10_metrics.png`

### Problema
Os nomes parecem legados e nÃ£o descrevem bem a finalidade do arquivo.

### RecomendaÃ§Ã£o
- `scenario_hexagonal_5x5.json`
- `federated_learning_metrics.png`

## 12) README e manifest.json

### Problema
O README e o manifest usam nomes diferentes para se comunicar sobre a mesma aplicaÃ§Ã£o.

### RecomendaÃ§Ã£o
Padronizar estes arquivos com um Ãºnico nome e uma Ãºnica narrativa.

Exemplo correto:
- `SimulatorFederatedLearning â€” CIFAR-10 com Dijkstra`

E remover referÃªncias a nomes histÃ³ricos ou variantes antigas.

## 13) PadronizaÃ§Ã£o de idioma

### Problema
O projeto mescla portuguÃªs e inglÃªs em textos de interface e documentaÃ§Ã£o.

### RecomendaÃ§Ã£o
Definir um padrÃ£o claro:
- Interface em portuguÃªs, para uso do usuÃ¡rio final
- Nomes tÃ©cnicos em inglÃªs, para classes e mÃ³dulos

Exemplos:
- `Carregar configuraÃ§Ã£o JSON` (portuguÃªs)
- `ExperimentConfig` (inglÃªs)
- `FLHistory` (inglÃªs)

## 14) Lista de renomeaÃ§Ãµes recomendadas

### Nome principal do sistema
- `SimulatorFederatedLearning`

### Nome tÃ©cnico de projeto
- `simulator_federated_learning`

### Nome do repositÃ³rio
- `SimulatorFederatedLearning`

### Classes
- `ImageMigrationClient` -> `CifarFederatedClient`
- `FlHistory` -> `FLHistory`
- `ComponentManager` -> `ScenarioExporter`

### ConfiguraÃ§Ã£o
- `delay_mode` -> `delay_mode`
- `fixed_weights` -> `fixed_delays`

### UI
- `SimulatorFederatedLearning â€” CIFAR-10 | Servidor Central + Dijkstra`
- `âš™ Carregar Config JSON` -> `âš™ Carregar configuraÃ§Ã£o JSON`
- `Treinar FL (Dijkstra)` -> `Treinar Modelo`
- `Ver MÃ©tricas FL` -> `Ver mÃ©tricas FL`
- `FL: NÃ£o treinado` -> `Status: NÃ£o treinado`

## 15) ConclusÃ£o

O projeto estÃ¡ funcional e bem estruturado em termos de arquitetura, mas a nomenclatura ainda apresenta inconsistÃªncias histÃ³ricas e de domÃ­nio que prejudicam clareza, manutenÃ§Ã£o e leitura. O maior problema Ã© a mistura de nomes do produto, do mÃ³dulo legado e do nome do repositÃ³rio.

A correÃ§Ã£o mais importante Ã© padronizar a identidade do software e depois ajustar classes, textos e artefatos exportados para refletirem esse padrÃ£o de forma consistente.

