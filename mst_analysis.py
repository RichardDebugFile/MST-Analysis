
"""
MST Analysis Toolkit
====================

Script de línea de comandos para comparar estrategias de Árbol de Cobertura Mínima (MST)
con NetworkX.

- Lee archivos CSV o JSON con la lista de aristas (source, target, weight)
- Construye el grafo y calcula el MST con Kruskal y Prim
- Muestra métricas (nº de aristas y peso total)
- Genera PNGs del grafo original y de cada MST

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CARGA Y CONSTRUCCIÓN DEL GRAFO
# ---------------------------------------------------------------------------

def load_edges(path: Path) -> pd.DataFrame:  # noqa: D401
    """Carga archivo CSV o JSON y devuelve un DataFrame con source, target, weight."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise ValueError("Formato no soportado: usa CSV o JSON")

    required = {"source", "target", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"El archivo debe tener las columnas {required}.")

    return df[list(required)]


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Crea un grafo no dirigido y ponderado a partir del DataFrame."""
    g = nx.Graph()
    for _, row in df.iterrows():
        g.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    return g

# ---------------------------------------------------------------------------
# MST Y ESTADÍSTICAS
# ---------------------------------------------------------------------------

def compute_mst(g: nx.Graph, algorithm: str = "kruskal") -> nx.Graph:
    """Devuelve el MST usando el algoritmo elegido ('kruskal' o 'prim')."""
    if algorithm not in {"kruskal", "prim"}:
        raise ValueError("algorithm debe ser 'kruskal' o 'prim'")
    return nx.minimum_spanning_tree(g, algorithm=algorithm, weight="weight")


def graph_stats(g: nx.Graph) -> Tuple[int, float]:
    """Devuelve (número de aristas, peso total)."""
    edges = g.number_of_edges()
    total_weight = sum(data["weight"] for _, _, data in g.edges(data=True))
    return edges, total_weight

# ---------------------------------------------------------------------------
# VISUALIZACIÓN
# ---------------------------------------------------------------------------

def draw_graph(g: nx.Graph, filepath: Path, title: str) -> None:
    """Guarda una imagen PNG del grafo."""
    pos = nx.spring_layout(g, seed=42)
    weights = nx.get_edge_attributes(g, "weight")

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(g, pos, node_size=300)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels={e: f"{w:.1f}" for e, w in weights.items()}
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:  
    parser = argparse.ArgumentParser(description="MST Analysis Toolkit")
    parser.add_argument("input_file", type=Path, help="Archivo CSV/JSON con aristas")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directorio de salida para las imágenes",
    )
    args = parser.parse_args()

    df = load_edges(args.input_file)
    graph = build_graph(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_graph(graph, args.output_dir / "graph_original.png", "Grafo original")

    for algo in ("kruskal", "prim"):
        mst = compute_mst(graph, algorithm=algo)
        n_edges, total_weight = graph_stats(mst)
        print(f"{algo.capitalize()} MST: {n_edges} aristas, peso total = {total_weight:.2f}")
        draw_graph(mst, args.output_dir / f"mst_{algo}.png", f"MST ({algo})")


if __name__ == "__main__":  
    main()
