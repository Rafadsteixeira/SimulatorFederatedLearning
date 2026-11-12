# -*- coding: utf-8 -*-
"""
SimulatorFederatedLearning — interface grafica (Tkinter).

O experimento em si fica em fl_engine.py; esta janela apenas carrega a
configuracao, dispara a execucao e mostra o mapa de rede e as metricas.
Para rodar sem interface: python run_experiment.py config.json
"""

import os
import sys
import tempfile
import threading
import urllib.request
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tkinter import ttk, messagebox, filedialog

import fl_engine as engine
from fl_engine import (
    BaseStation,
    CENTRAL_BS,
    CENTRAL_SW,
    CLASS_COLORS,
    ExperimentConfig,
    NUM_ROUNDS_FL,
    RESULTS_DIR,
    Topology,
    client_stations,
    dijkstra_distances,
    load_cifar10,
    load_sample_images_for_map,
    run_federated_training_image,
)

config_file_name: str = ""

# ==============================================================================
#  CONFIGURACAO VISUAL DO GRAFO
# ==============================================================================

positions:        Dict = {}
node_labels:      Dict = {}
base_sizes:       List = []
base_colors:      List = []
edge_colors_list: List = []
edge_widths_list: List = []


def setup_graph_style():
    global positions, node_labels, base_sizes, base_colors
    global edge_colors_list, edge_widths_list

    topo  = Topology.first()
    max_d = max(dijkstra_distances.values()) if dijkstra_distances else 1.0

    positions   = {}
    node_labels = {}
    base_sizes  = []
    base_colors = []

    for node in topo.nodes():
        positions[node]   = node.coordinates
        if node == CENTRAL_SW:
            node_labels[node] = f"*\nS{node.id}"
            base_sizes.append(1100)
            base_colors.append("#FFD700")
        else:
            node_labels[node] = f"C{node.id}"
            base_sizes.append(600)
            d     = dijkstra_distances.get(node, max_d)
            ratio = d / max_d
            r = int(30  + ratio * 80)
            g = int(100 - ratio * 60)
            b = int(220 - ratio * 80)
            base_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    edge_colors_list = []
    edge_widths_list = []
    for (u, v) in topo.edges():
        delay     = topo.edges[u, v]['delay']
        bandwidth = topo.edges[u, v]['bandwidth']
        cost = delay / bandwidth
        if cost < 0.1:
            edge_colors_list.append('#32CD32')
            edge_widths_list.append(1.8)
        elif cost < 0.8:
            edge_colors_list.append('#FFA500')
            edge_widths_list.append(2.2)
        else:
            edge_colors_list.append('#FF4444')
            edge_widths_list.append(2.8)


setup_graph_style()

# ==============================================================================
#  GUI - JANELA PRINCIPAL
# ==============================================================================

window = tk.Tk()
window.title(
    "SimulatorFederatedLearning"
)
window.geometry("1480x920")
window.configure(bg="#1A1A2E")


def on_closing():
    window.quit()
    window.destroy()
    sys.exit(0)


window.protocol("WM_DELETE_WINDOW", on_closing)

style = ttk.Style()
try:
    style.theme_use("clam")
except Exception:
    pass
style.configure("TFrame",           background="#1A1A2E")
style.configure("TLabel",           background="#1A1A2E", foreground="#E0E0E0")
style.configure("TButton",          background="#2A2A5A", foreground="#E0E0E0")
style.configure("TCombobox",        fieldbackground="#2A2A5A", foreground="#E0E0E0")
style.configure("TSeparator",       background="#3A3A6A")
style.configure("TLabelframe",      background="#1A1A2E", foreground="#E0E0E0")
style.configure("TLabelframe.Label",background="#1A1A2E", foreground="#FFD700")

ctrl_frame = ttk.Frame(window, padding="6")
ctrl_frame.pack(fill=tk.X, side=tk.TOP)

fl_status_var = tk.StringVar(value=f"Status: {engine.fl_training_status}")


def update_fl_status(text: str):
    fl_status_var.set(f"Status: {text}")
    window.update_idletasks()


def _apply_config(path: str, display_name: str):
    global config_file_name
    try:
        cfg = ExperimentConfig.from_json(path)
        engine.active_config    = cfg
        config_file_name = display_name
        _refresh_config_panel()
        messagebox.showinfo("Configuracao carregada",
                            f"Fonte: {config_file_name}\n"
                            + "\n".join(cfg.to_summary_lines()))
    except Exception as e:
        messagebox.showerror("Erro ao carregar arquivo JSON", str(e))


def _load_from_url_dialog():
    url_win = tk.Toplevel(window)
    url_win.title("Carregar Parametros")
    url_win.geometry("500x160")
    url_win.configure(bg="#1A1A2E")
    url_win.resizable(False, False)
    url_win.grab_set()

    ttk.Label(url_win, text="URL do arquivo JSON:",
              foreground="#E0E0E0").pack(pady=(18, 4))
    url_entry = ttk.Entry(url_win, width=58)
    url_entry.pack(padx=20)
    url_entry.insert(0, "https://")

    def _fetch():
        url = url_entry.get().strip()
        if not url or url == "https://":
            return
        url_win.destroy()
        try:
            update_fl_status("Baixando JSON...")
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                             delete=False, encoding='utf-8') as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            name = url.rstrip("/").split("/")[-1] or "url_config.json"
            _apply_config(tmp_path, name)
            os.unlink(tmp_path)
            update_fl_status("Config URL carregada")
        except Exception as e:
            messagebox.showerror("Erro ao baixar JSON", str(e))
            update_fl_status("Erro no download")

    btn_f = ttk.Frame(url_win)
    btn_f.pack(pady=14)
    ttk.Button(btn_f, text="Carregar", command=_fetch).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_f, text="Cancelar", command=url_win.destroy).pack(side=tk.LEFT, padx=10)


def load_config_json():
    choice_win = tk.Toplevel(window)
    choice_win.title("Carregar Configuracao JSON")
    choice_win.geometry("380x170")
    choice_win.configure(bg="#1A1A2E")
    choice_win.resizable(False, False)
    choice_win.grab_set()

    ttk.Label(choice_win, text="Como deseja carregar o JSON?",
              font=("Arial", 11, "bold"), foreground="#FFD700",
              background="#1A1A2E").pack(pady=(20, 12))

    def _file():
        choice_win.destroy()
        path = filedialog.askopenfilename(
            title="Selecionar configuracao JSON",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )
        if path:
            _apply_config(path, os.path.basename(path))

    def _url():
        choice_win.destroy()
        _load_from_url_dialog()

    btn_f = ttk.Frame(choice_win)
    btn_f.pack(pady=4)
    ttk.Button(btn_f, text="Arquivo Local",
               command=_file, width=18).pack(side=tk.LEFT, padx=10)
    ttk.Button(btn_f, text="URL / Link",
               command=_url, width=18).pack(side=tk.LEFT, padx=10)
    ttk.Button(choice_win, text="Cancelar",
               command=choice_win.destroy).pack(pady=8)


def _refresh_config_panel():
    if engine.active_config is None:
        cfg_text_var.set("Nenhum arquivo de configuracao carregado.\nUsando valores padrao.")
    else:
        lines = [f"Fonte: {config_file_name}"] + engine.active_config.to_summary_lines()
        cfg_text_var.set("\n".join(lines))


def _safe_after(func, *args):
    """Agenda func no loop do Tk; ignora se a janela ja foi fechada."""
    try:
        window.after(0, func, *args)
    except Exception:
        pass


def start_fl_training():
    if engine.active_config is None:
        defaults = ExperimentConfig()
        ok = messagebox.askyesno(
            "Sem configuracao JSON",
            f"Nenhum JSON carregado.\n"
            f"Usar padroes: {defaults.n_clients} clientes, "
            f"{defaults.n_rounds} rounds, modo '{defaults.weight_mode}'.\n\n"
            "Deseja continuar?"
        )
        if not ok:
            return

    if engine._cifar_trainset is None:
        messagebox.showinfo("Atencao",
                            "Carregue o dataset CIFAR-10 primeiro\n"
                            "(botao 'Carregar Dataset').")
        return
    fl_btn.config(state=tk.DISABLED)
    metrics_btn.config(state=tk.DISABLED)

    def _thread():
        run_federated_training_image(
            status_callback=update_fl_status,
            round_callback=lambda _round, entry: _safe_after(_gui_append_round, entry),
            on_start=lambda: _safe_after(_gui_clear_round_list),
            on_finish=lambda: _safe_after(refresh_map),
        )
        window.after(0, lambda: fl_btn.config(state=tk.NORMAL))
        window.after(0, lambda: metrics_btn.config(state=tk.NORMAL))

    threading.Thread(target=_thread, daemon=True).start()


def start_load_dataset():
    load_btn.config(state=tk.DISABLED)
    update_fl_status("Baixando CIFAR-10...")

    def _thread():
        load_cifar10()
        load_sample_images_for_map()
        update_fl_status("Dataset pronto")
        window.after(0, refresh_map)
        window.after(0, lambda: load_btn.config(state=tk.NORMAL))

    threading.Thread(target=_thread, daemon=True).start()


load_btn = ttk.Button(ctrl_frame, text="Carregar Dataset",
                      command=start_load_dataset)
load_btn.pack(side=tk.LEFT, padx=4)

fl_btn = ttk.Button(ctrl_frame, text="Treinar Modelo",
                    command=start_fl_training)
fl_btn.pack(side=tk.LEFT, padx=4)

ttk.Label(ctrl_frame, textvariable=fl_status_var,
          foreground="#9ECDFF").pack(side=tk.LEFT, padx=6)

ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

metrics_btn = ttk.Button(ctrl_frame, text="Ver metricas",
                         command=lambda: show_fl_metrics())
metrics_btn.pack(side=tk.LEFT, padx=4)

ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

json_btn = ttk.Button(ctrl_frame, text="Configurações",
                      command=load_config_json)
json_btn.pack(side=tk.LEFT, padx=4)

content_frame = ttk.Frame(window)
content_frame.pack(fill=tk.BOTH, expand=True)

map_frame = ttk.Frame(content_frame)
map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

side_frame = ttk.Frame(content_frame, width=380)
side_frame.pack(side=tk.RIGHT, fill=tk.Y)
side_frame.pack_propagate(False)

fig_map, ax_map = plt.subplots(figsize=(11, 7.5))
fig_map.patch.set_facecolor("#0F0F1A")
ax_map.set_facecolor("#0F0F1A")

canvas_map = FigureCanvasTkAgg(fig_map, master=map_frame)
canvas_map.draw()
canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

cfg_panel = ttk.LabelFrame(side_frame, text="Configuracao Ativa", padding=6)
cfg_panel.pack(fill=tk.X, padx=8, pady=(8, 2))
cfg_text_var = tk.StringVar(
    value="Nenhum arquivo de configuracao carregado.\nUsando valores padrao."
)
ttk.Label(cfg_panel, textvariable=cfg_text_var, justify=tk.LEFT,
          foreground="#B0BEC5", font=("Consolas", 8)).pack(fill=tk.X)

rounds_panel = ttk.LabelFrame(side_frame, text="Rodadas de Treinamento", padding=6)
rounds_panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))

_rounds_list_frame = ttk.Frame(rounds_panel)
_rounds_list_frame.pack(fill=tk.BOTH, expand=True)

rounds_listbox = tk.Listbox(
    _rounds_list_frame,
    bg="#0F0F1A", fg="#9ECDFF",
    selectbackground="#2A4A8A", selectforeground="#FFFFFF",
    font=("Consolas", 9),
    activestyle="none",
    borderwidth=0, highlightthickness=0,
    relief=tk.FLAT,
)
rounds_scrollbar = ttk.Scrollbar(_rounds_list_frame, orient=tk.VERTICAL,
                                  command=rounds_listbox.yview)
rounds_listbox.configure(yscrollcommand=rounds_scrollbar.set)
rounds_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
rounds_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

details_panel = ttk.LabelFrame(side_frame, text="Detalhes da Rodada", padding=8)
details_panel.pack(fill=tk.X, padx=8, pady=(0, 4))

details_text_var = tk.StringVar(value="Clique em uma rodada para\nver os detalhes.")
details_label = ttk.Label(
    details_panel,
    textvariable=details_text_var,
    justify=tk.LEFT,
    wraplength=340,
    foreground="#CFD8DC",
    font=("Consolas", 9),
)
details_label.pack(fill=tk.X)

legend_frame = ttk.LabelFrame(side_frame, text="Legenda", padding=6)
legend_frame.pack(fill=tk.X, padx=8, pady=4)

legend_items = [
    ("*  Servidor Central", "#FFD700"),
    ("o  Cliente (perto)", "#3060DC"),
    ("o  Cliente (longe)",  "#5A2A8A"),
    ("-- Link rapido",      "#32CD32"),
    ("-- Link medio",       "#FFA500"),
    ("-- Link lento",       "#FF4444"),
]
for text, color in legend_items:
    lf = ttk.Frame(legend_frame)
    lf.pack(fill=tk.X, pady=1)
    tk.Label(lf, text="=", foreground=color,
             background="#1A1A2E", font=("Arial", 11)).pack(side=tk.LEFT)
    ttk.Label(lf, text=text, foreground="#CFD8DC",
              font=("Arial", 9)).pack(side=tk.LEFT, padx=4)

_selected_bs: Optional[BaseStation] = None


# ==============================================================================
#  DESENHO DO MAPA DE REDE
# ==============================================================================

def draw_network_map(highlight_bs: Optional[BaseStation] = None,
                     highlight_path: Optional[List] = None):
    topo = Topology.first()
    ax_map.cla()
    ax_map.set_facecolor("#0F0F1A")

    max_d = max(dijkstra_distances.values()) if dijkstra_distances else 1.0
    node_colors = []
    for node in topo.nodes():
        if node == CENTRAL_SW:
            node_colors.append("#FFD700")
        elif highlight_bs and node == highlight_bs.network_switch:
            node_colors.append("#00FFFF")
        else:
            d     = dijkstra_distances.get(node, max_d)
            ratio = d / max_d
            r = int(30  + ratio * 80)
            g = int(100 - ratio * 60)
            b = int(220 - ratio * 80)
            node_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    nx.draw_networkx_edges(
        topo, ax=ax_map, pos=positions,
        edge_color=edge_colors_list,
        width=edge_widths_list,
        alpha=0.7
    )
    nx.draw_networkx_nodes(
        topo, ax=ax_map, pos=positions,
        node_color=node_colors,
        node_size=base_sizes,
        alpha=0.9
    )

    if highlight_path and len(highlight_path) >= 2:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(
            topo, ax=ax_map, pos=positions,
            edgelist=path_edges,
            edge_color="#00FF88",
            width=4.0,
            style="solid",
            alpha=1.0
        )

    if engine._images_loaded and engine._client_sample_images:
        for bs in client_stations:
            if bs.id in engine._client_sample_images:
                img_arr, cls_name, cls_idx = engine._client_sample_images[bs.id]
                node = bs.network_switch
                if node in positions:
                    zoom = 0.48 if node != (highlight_bs.network_switch
                                            if highlight_bs else None) else 0.62
                    imagebox = OffsetImage(img_arr, zoom=zoom)
                    imagebox.image.axes = ax_map
                    border_color = ("#00FFFF" if highlight_bs and
                                    node == highlight_bs.network_switch
                                    else CLASS_COLORS[cls_idx])
                    ab = AnnotationBbox(
                        imagebox, positions[node],
                        frameon=True,
                        bboxprops=dict(edgecolor=border_color,
                                       linewidth=2, boxstyle="round,pad=0.1"),
                        pad=0.05,
                        xycoords='data'
                    )
                    ax_map.add_artist(ab)

    cx, cy = CENTRAL_SW.coordinates
    ax_map.scatter(cx, cy, c="#FFD700", s=800, zorder=10,
                   edgecolors="white", linewidths=2)
    ax_map.text(cx, cy + 0.25, f"* Servidor\nBS{CENTRAL_BS.id}",
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                color="#FFD700")

    if not engine._images_loaded:
        for node in topo.nodes():
            if node != CENTRAL_SW:
                x, y = positions[node]
                ax_map.text(x, y, f"C{node.id}", ha='center', va='center',
                            fontsize=6, color='white', fontweight='bold')

    ax_map.set_title(
        "Rede Hexagonal 5x5 - SimulatorFederatedLearning com CIFAR-10\n"
        f"Servidor Central: BS{CENTRAL_BS.id} | "
        f"Clientes: {len(client_stations)} | "
        f"Strategy: TopologyAwareFedAvg",
        color="#E0E0E0", fontsize=11, pad=10
    )
    ax_map.axis('off')
    canvas_map.draw()


def draw_initial():
    draw_network_map()


# ==============================================================================
#  CALLBACKS DA GUI
# ==============================================================================

def _gui_clear_round_list():
    rounds_listbox.delete(0, tk.END)
    details_text_var.set("Clique em uma rodada para\nver os detalhes.")


def _gui_append_round(entry: Dict):
    r    = entry["round"]
    acc  = entry["val_acc"] * 100
    loss = entry["val_loss"]
    n    = entry["n_clients"]
    mode = entry.get("weight_mode", "?")[0].upper()
    hl   = " HL" if entry.get("is_hl_round") else ""
    line = f" R{r:>2} [{mode}]{hl}  Loss {loss:.4f}  Acc {acc:5.1f}%  C:{n}"
    rounds_listbox.insert(tk.END, line)
    if entry.get("is_hl_round"):
        color = "#3A1A0A"
    else:
        color = "#1A2A4A" if r % 2 == 0 else "#0F1A30"
    rounds_listbox.itemconfig(tk.END, background=color)
    rounds_listbox.selection_clear(0, tk.END)
    rounds_listbox.selection_set(tk.END)
    rounds_listbox.see(tk.END)
    _show_round_details(len(engine.fl_round_log) - 1)


def _show_round_details(idx: int):
    if idx < 0 or idx >= len(engine.fl_round_log):
        details_text_var.set("Nenhum dado disponivel.")
        return
    e        = engine.fl_round_log[idx]
    r        = e["round"]
    n_cli    = e["n_clients"]
    samples  = e["total_samples"]
    tl       = e["train_loss"]
    ta       = e["train_acc"] * 100
    vl       = e["val_loss"]
    va       = e["val_acc"] * 100

    delta_loss_str = ""
    delta_acc_str  = ""
    if idx > 0:
        prev   = engine.fl_round_log[idx - 1]
        d_loss = vl - prev["val_loss"]
        d_acc  = va - prev["val_acc"] * 100
        sign_l = "+" if d_loss >= 0 else ""
        sign_a = "+" if d_acc  >= 0 else ""
        delta_loss_str = f"  ({sign_l}{d_loss:.4f})"
        delta_acc_str  = f"  ({sign_a}{d_acc:.2f}%)"

    total_rounds = (engine.active_config.n_rounds if engine.active_config else NUM_ROUNDS_FL)
    mode  = e.get("weight_mode", "?").upper()
    is_hl = e.get("is_hl_round", False)
    sep   = "=" * 33

    deadline_s = e.get("deadline", 0.0)
    on_time_c  = e.get("clients_on_time", n_cli)
    late_c     = e.get("clients_late", 0)
    avg_t      = e.get("avg_time", 0.0)
    up_norm    = e.get("update_norm", 0.0)

    lines = [
        f"+{sep}+",
        f"  Rodada  : {r} / {total_rounds}  [{mode}]{'  HL' if is_hl else ''}",
        f"  Clientes: {n_cli} (No prazo: {on_time_c} | Atrasados: {late_c})",
        f"  Deadline: {deadline_s:.2f}s | T.Med: {avg_t:.2f}s",
        f"  Amostras: {samples:,}",
        f"+{sep}+",
        f"  TREINO  (FedAvg ponderado)",
        f"  Loss    : {tl:.4f}",
        f"  Acc     : {ta:.2f}%",
        f"  ||Δw||  : {up_norm:.3f} (norma media)",
        f"+{sep}+",
        f"  VALIDACAO",
        f"  Loss    : {vl:.4f}{delta_loss_str}",
        f"  Acc     : {va:.2f}%{delta_acc_str}",
        f"+{sep}+",
        f"  CLIENTES (Tempo | Prazo | Inform.)",
    ]

    for cd in e.get("clients_detail", []):
        cid     = cd["cid"]
        bs_id   = cd["bs_id"]
        t_tot   = cd.get("t_total")
        on_tm   = cd.get("on_time")
        tau     = cd.get("staleness", 0)
        c_loss  = cd.get("loss")
        u_norm  = cd.get("update_norm")
        status  = cd.get("status")
        status_labels = {
            "on_time": "OK",
            "late_pending": "PENDENTE",
            "accepted_stale": f"STALE(tau={tau})",
            "accepted_reduced": "PESO REDUZIDO",
            "discarded_deadline": "DESCARTADO (prazo)",
            "discarded_staleness": f"DESCARTADO (tau={tau})",
            "failed": "FALHOU",
        }
        st_tag = status_labels.get(
            status,
            "LATE" if on_tm is False else "OK" if on_tm is True else "DESCONHECIDO",
        )
        time_label = f"{float(t_tot):4.1f}s" if t_tot is not None else "   -- "
        norm_label = f"{float(u_norm):4.2f}" if u_norm is not None else "  --"
        loss_label = f"{float(c_loss):4.2f}" if c_loss is not None else "  --"
        lines.append(
            f"  [C{cid:02d}|BS{bs_id:<2}] t={time_label} [{st_tag}]"
            f" ||dw||={norm_label} L={loss_label}"
        )

    lines.append(f"+{sep}+")
    details_text_var.set("\n".join(lines))


def _on_round_select(event):
    sel = rounds_listbox.curselection()
    if sel:
        _show_round_details(sel[0])


rounds_listbox.bind("<<ListboxSelect>>", _on_round_select)


def refresh_map():
    draw_network_map()


# ==============================================================================
#  JANELA DE METRICAS FL
# ==============================================================================

def show_fl_metrics():
    if engine.fl_history is None or not engine.fl_history.losses_distributed:
        messagebox.showinfo(
            "Metricas FL",
            "Nenhuma metrica disponivel.\n"
            "Execute o treinamento FL primeiro."
        )
        return

    met_win = tk.Toplevel(window)
    met_win.title("Metricas - SimulatorFederatedLearning CIFAR-10")
    met_win.geometry("1000x680")
    met_win.configure(bg="#1A1A2E")

    fig_m, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig_m.patch.set_facecolor("#1A1A2E")
    fig_m.suptitle(
        f"SimulatorFederatedLearning - CIFAR-10 | {engine.fl_history.num_rounds} rounds | "
        f"{engine.fl_history.num_clients} clientes | TopologyAwareFedAvg + Dijkstra",
        fontsize=12, fontweight='bold', color='#E0E0E0'
    )

    rounds_e = [r for r, _ in engine.fl_history.losses_distributed]
    losses_e = [l for _, l in engine.fl_history.losses_distributed]

    def _style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#0F0F1A")
        ax.set_title(title, color='#E0E0E0', fontsize=10)
        ax.set_xlabel(xlabel, color='#9E9E9E', fontsize=9)
        ax.set_ylabel(ylabel, color='#9E9E9E', fontsize=9)
        ax.tick_params(colors='#9E9E9E')
        ax.grid(True, alpha=0.2, color='#333355')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    axes[0, 0].plot(rounds_e, losses_e, 'o-', color='#4FC3F7',
                    linewidth=2, markersize=7, label="Val Loss")
    axes[0, 0].fill_between(rounds_e, losses_e, alpha=0.15, color='#4FC3F7')
    _style_ax(axes[0, 0], "Loss de Validacao por Round", "Round", "Loss")
    axes[0, 0].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    if 'accuracy' in engine.fl_history.eval_metrics:
        acc_rounds = [r for r, _ in engine.fl_history.eval_metrics['accuracy']]
        acc_vals   = [v * 100 for _, v in engine.fl_history.eval_metrics['accuracy']]
        axes[0, 1].plot(acc_rounds, acc_vals, 's-', color='#66BB6A',
                        linewidth=2, markersize=7, label="Val Acc")
        axes[0, 1].fill_between(acc_rounds, acc_vals, alpha=0.15, color='#66BB6A')
        axes[0, 1].set_ylim(0, 100)
        _style_ax(axes[0, 1], "Acuracia de Validacao por Round", "Round", "Acuracia (%)")
        axes[0, 1].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    if 'train_loss' in engine.fl_history.fit_metrics:
        tl_r = [r for r, _ in engine.fl_history.fit_metrics['train_loss']]
        tl_v = [v for _, v in engine.fl_history.fit_metrics['train_loss']]
        axes[1, 0].plot(tl_r, tl_v, 'D-', color='#FFA726',
                        linewidth=2, markersize=7, label="Train Loss")
        axes[1, 0].fill_between(tl_r, tl_v, alpha=0.15, color='#FFA726')
        _style_ax(axes[1, 0], "Train Loss (FedAvg Agregado)", "Round", "Loss")
        axes[1, 0].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    if 'train_acc' in engine.fl_history.fit_metrics:
        ta_r = [r for r, _ in engine.fl_history.fit_metrics['train_acc']]
        ta_v = [v * 100 for _, v in engine.fl_history.fit_metrics['train_acc']]
        axes[1, 1].plot(ta_r, ta_v, '^-', color='#CE93D8',
                        linewidth=2, markersize=7, label="Train Acc")
        axes[1, 1].fill_between(ta_r, ta_v, alpha=0.15, color='#CE93D8')
        axes[1, 1].set_ylim(0, 100)
        _style_ax(axes[1, 1], "Train Accuracy (Agregada)", "Round", "Acuracia (%)")
        axes[1, 1].legend(facecolor="#1A1A2E", labelcolor="#E0E0E0")

    plt.tight_layout()
    fig_m.savefig(os.path.join(RESULTS_DIR, "fl_cifar10_metrics.png"),
                  dpi=120, bbox_inches="tight", facecolor=fig_m.get_facecolor())
    print(f"  Grafico salvo -> {RESULTS_DIR}/fl_cifar10_metrics.png")

    canvas_m = FigureCanvasTkAgg(fig_m, master=met_win)
    canvas_m.draw()
    canvas_m.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    sum_frame = ttk.LabelFrame(met_win, text="Resumo Final", padding=6)
    sum_frame.pack(fill=tk.X, padx=10, pady=6)

    final_r    = engine.fl_history.losses_distributed[-1][0]
    final_loss = engine.fl_history.losses_distributed[-1][1]
    final_acc  = (engine.fl_history.eval_metrics['accuracy'][-1][1] * 100
                  if 'accuracy' in engine.fl_history.eval_metrics else 0)

    ttk.Label(
        sum_frame,
        text=(f"Round {final_r}  |  Val Loss: {final_loss:.4f}  |  "
              f"Val Acc: {final_acc:.2f}%  |  "
              f"Clientes: {engine.fl_history.num_clients}  |  "
              f"Strategy: TopologyAwareFedAvg + Dijkstra"),
        font=('Arial', 10, 'bold')
    ).pack()

    def _close():
        plt.close(fig_m)
        met_win.destroy()

    met_win.protocol("WM_DELETE_WINDOW", _close)


# ==============================================================================
#  INICIALIZACAO
# ==============================================================================

if __name__ == "__main__":
    draw_initial()

    print("  Interface carregada. Use os botoes para:")
    print("  1. 'Carregar CIFAR-10'          -> baixa dataset e exibe thumbnails nos nos")
    print("  2. 'Treinar Modelo'             -> executa rounds via Flower (ClientApp + ServerApp)")
    print("  3. 'Ver metricas FL'            -> graficos de convergencia")
    print("  4. 'Carregar configuracao JSON' -> personaliza o experimento\n")

    window.mainloop()
