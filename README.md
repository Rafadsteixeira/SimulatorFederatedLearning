# SimulatorFederatedLearning

Projeto reestruturado para uma baseline mínima baseada em Flower, com runtime real de treinamento federado usando `Strategy`, `ServerApp` e `ClientApp`.

## Visão geral

O objetivo agora é manter uma implementação enxuta e compatível com o Flower moderno, usando FedAvg como estratégia principal e CIFAR-10 particionado por cliente como conjunto de dados de treinamento.

- `Strategy`: agregação federada via FedAvg
- `ServerApp`: servidor do runtime Flower
- `ClientApp`: clientes locais executando treino em partes do dataset
- `models/federated/`: arquitetura do modelo e utilitários de agregação

## Estrutura atual

```text
SimulatorFederatedLearning/
├── flower/
│   ├── __init__.py
│   ├── baseline.py
│   ├── client.py
│   ├── server.py
│   └── strategy.py
├── models/
│   └── federated/
│       ├── __init__.py
│       ├── cifar_cnn.py
│       └── fl_aggregator.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Como executar

```bash
python main.py
```

A execução cria uma simulação em memória com vários clientes, particiona o CIFAR-10, treina localmente em cada nó e agrega os pesos com FedAvg usando o runtime do Flower.

## Dependências

```bash
pip install -r requirements.txt
pip install ray
```

O pacote `ray` é necessário para o backend de simulação do Flower no ambiente local.

## Observação

A arquitetura foi simplificada para manter apenas a base do projeto orientada ao Flower. O código atual foca em um baseline funcional e sustentável, sem o acoplamento antigo dos protótipos de UI e simulação legada.
