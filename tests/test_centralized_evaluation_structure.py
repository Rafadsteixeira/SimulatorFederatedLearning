import ast
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE = (PROJECT_DIR / "SimulatorFederatedLearning.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def class_methods(class_name):
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Classe ausente: {class_name}")


def class_source(class_name):
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(SOURCE, node)
    raise AssertionError(f"Classe ausente: {class_name}")


def test_clients_do_not_expose_repeated_global_evaluation():
    methods = class_methods("CifarFederatedClient")
    client_source = class_source("CifarFederatedClient")

    assert "fit" in methods
    assert "evaluate" not in methods
    assert "testloader" not in client_source


def test_strategy_owns_centralized_evaluation():
    methods = class_methods("TopologyAwareFedAvg")

    assert "evaluate_and_record_round" in methods
    assert "configure_evaluate" in methods
    assert '"evaluation_mode": "centralized"' in SOURCE
    assert "cli.evaluate(" not in SOURCE


def test_global_evaluation_helper_exists():
    functions = {
        node.name for node in TREE.body if isinstance(node, ast.FunctionDef)
    }

    assert "evaluate_global_weights" in functions
