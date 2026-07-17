# -*- coding: utf-8 -*-
"""
main.py — Ponto de entrada da aplicação migration_fdl.

Uso:
    python main.py

Fluxo:
    1. Instala dependências (via pip, silencioso).
    2. Instancia o AppController.
    3. Inicia a aplicação (topologia + GUI).
"""
import sys
import os

# Força UTF-8 no console do Windows (evita UnicodeEncodeError com → ✓ ⚠ etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Garante que o diretório raiz do projeto está no sys.path,
# permitindo imports absolutos de models/, views/, controllers/, utils/.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --- Backend matplotlib: deve ser definido UMA VEZ, antes de qualquer outro
#     import de matplotlib.pyplot (inclusive nas views) ---
import matplotlib
matplotlib.use("TkAgg")


# --- 1. Instalação silenciosa de dependências ---
from utils.installer import ensure_dependencies
ensure_dependencies()

# --- 2. Importações do projeto (após instalação) ---
from controllers.app_controller import AppController


def main() -> None:
    """Inicializa e executa a aplicação."""
    app = AppController()
    app.start()


if __name__ == "__main__":
    main()


