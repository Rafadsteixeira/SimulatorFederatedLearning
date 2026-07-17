# -*- coding: utf-8 -*-
"""
utils/installer.py
Instalação automática de dependências via pip durante o startup.
"""
import subprocess
import sys

REQUIRED_PACKAGES = [
    "matplotlib",
    "numpy",
    "networkx",
    "torch",
    "torchvision",
    "flwr",
    "Pillow",
]


def ensure_dependencies(packages: list = None, silent: bool = True) -> None:
    """
    Tenta instalar cada pacote listado via pip.

    Args:
        packages: Lista de pacotes a verificar/instalar.
                  Usa REQUIRED_PACKAGES por padrão.
        silent:   Se True, suprime stdout/stderr do pip.
    """
    if packages is None:
        packages = REQUIRED_PACKAGES

    for pkg in packages:
        try:
            kwargs = {}
            if silent:
                kwargs = {
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                **kwargs,
            )
        except Exception:
            print(f"Aviso: falha ao instalar '{pkg}', tentando continuar...")
