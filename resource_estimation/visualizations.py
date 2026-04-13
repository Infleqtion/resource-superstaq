from itertools import chain

import cirq
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
from . import lattice_surgery_primitives as lsp
from .layout import Layout


def visualize_layout_moment(
    G: nx.Graph, moment_paths: list[list[str]], column_layout: Layout
):  # pragma: no cover
    """
    This probably does not work anymore without a significant amount of changes.
    """
    moment_paths_flat = list(chain.from_iterable(moment_paths))
    diction = {}
    for qubit in column_layout.qubits:
        diction[qubit["name"]] = (qubit["qubit"].col, -qubit["qubit"].row)
        if qubit["type"] == "qubit":
            G.nodes[qubit["name"]]["color"] = "red"
        elif qubit["type"] == "factory":
            G.nodes[qubit["name"]]["color"] = "green"
        else:
            G.nodes[qubit["name"]]["color"] = "cyan"
    for edge in G.edges:
        G.edges[edge]["color"] = "black"
    distinct_moments = [[]]
    for path in moment_paths:
        inserted = False
        for i in range(0, len(distinct_moments)):
            conflict = False
            for j in range(0, len(distinct_moments[i])):
                if has_intersection(distinct_moments[i][j], path):
                    conflict = True
            if not conflict:
                inserted = True
                distinct_moments[i].append(path)
                break
        if not inserted:
            distinct_moments.append([path])
    fig, axes = plt.subplots(
        nrows=len(distinct_moments),
        ncols=1,
        figsize=(7, len(distinct_moments) * column_layout.rows),
    )
    for i in range(0, len(distinct_moments)):
        for path in distinct_moments[i]:
            for j in range(0, len(path) - 1):
                if path[j][0] == "a":
                    G.nodes[path[j]]["color"] = "yellow"
                G.edges[path[j], path[j + 1]]["color"] = "yellow"
        node_colors = [G.nodes[node]["color"] for node in G.nodes()]
        edge_colors = [G.edges[edge]["color"] for edge in G.edges()]
        nx.draw(
            G,
            pos=diction,
            ax=axes[i] if len(distinct_moments) > 1 else axes,
            node_color=node_colors,
            edge_color=edge_colors,
            with_labels=True,
        )
        for qubit in column_layout.qubits:
            if G.nodes[qubit["name"]]["color"] == "yellow":
                G.nodes[qubit["name"]]["color"] = "cyan"

        for edge in G.edges:
            if G.edges[edge]["color"] == "yellow":
                G.edges[edge]["color"] = "black"
    plt.show()


def display_NN_graph(G: nx.Graph):  # pragma: no cover
    """
    This can display the connectivity graph of a layout effectively
    """
    pos_dict = {}
    labels = {}
    node_colors = []
    for grid_qubit in list(G.nodes):
        assert type(grid_qubit) == cirq.GridQubit
        pos_dict[grid_qubit] = (grid_qubit.col, -1 * grid_qubit.row)
        if G.nodes[grid_qubit]["patch_type"] == "factory":
            labels[grid_qubit] = G.nodes[grid_qubit]["ftype"]
            node_colors.append("green")
        elif G.nodes[grid_qubit]["patch_type"] == "ancilla":
            labels[grid_qubit] = "a"
            node_colors.append("cyan")
        elif G.nodes[grid_qubit]["patch_type"] == "data":
            labels[grid_qubit] = "D"
            node_colors.append("red")
        elif G.nodes[grid_qubit]["patch_type"] == "distill":
            labels[grid_qubit] = "f"
            node_colors.append("brown")
    nx.draw(
        G,
        pos=pos_dict,
        with_labels=True,
        font_size=12,
        labels=labels,
        node_color=node_colors,
    )
    plt.show()


def display_move_moments(ops: list[list[cirq.Operation]], lay: Layout):  # pragma: no cover
    """
    Little animation for tracking factory usage in the (slow) movement layout. The new fast
    layouts don't really use graphs
    """
    G = lay.layout_graph
    pos_dict = {}
    labels = {}
    for grid_qubit in list(G.nodes):
        pos_dict[grid_qubit] = (grid_qubit.col, -1 * grid_qubit.row)
        if G.nodes[grid_qubit]["patch_type"] == "factory":
            labels[grid_qubit] = G.nodes[grid_qubit]["ftype"]
            G.nodes[grid_qubit]["color"] = "green"
        if G.nodes[grid_qubit]["patch_type"] == "data":
            labels[grid_qubit] = "d"
            G.nodes[grid_qubit]["color"] = "red"

    def animate(i):
        moment_ops = ops[i]
        G = lay.layout_graph
        for edge in G.edges:
            G.edges[edge]["color"] = "black"
        for op in moment_ops:
            if op.gate in cirq.GateFamily(cirq.CNOT):
                if G.nodes[op.qubits[0]]["patch_type"] == "factory":
                    G.nodes[op.qubits[0]]["color"] = "gray"
                G.edges[(op.qubits[0], op.qubits[1])]["color"] = "yellow"
            elif op.gate in cirq.GateFamily(lsp.Cultivate):
                G.nodes[op.qubits[0]]["color"] = "green"
        node_colors = []
        edge_colors = []
        for node in list(G.nodes):
            node_colors.append(G.nodes[node]["color"])
        for edge in list(G.edges):
            edge_colors.append(G.edges[edge]["color"])
        nx.draw(
            G,
            pos=pos_dict,
            with_labels=True,
            font_size=12,
            labels=labels,
            node_color=node_colors,
            edge_color=edge_colors,
        )

    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(
        fig, animate, frames=len(ops), interval=1000, blit=False, repeat=False
    )
    plt.show()


def display_lattice_moments(ops: list[list[cirq.Operation]], lay: Layout):  # pragma: no cover
    """
    Little animation for tracking factory usage in the (slow) lattice surgery layout. The new fast
    layouts don't really use graphs, maybe there is some way to change these to bring it back though
    """
    G = lay.layout_graph
    pos_dict = {}
    labels = {}
    for grid_qubit in list(G.nodes):
        pos_dict[grid_qubit] = (grid_qubit.col, -1 * grid_qubit.row)
        if G.nodes[grid_qubit]["patch_type"] == "factory":
            labels[grid_qubit] = G.nodes[grid_qubit]["ftype"]
            G.nodes[grid_qubit]["color"] = "green"
        if G.nodes[grid_qubit]["patch_type"] == "data":
            labels[grid_qubit] = "d"
            G.nodes[grid_qubit]["color"] = "red"
        if G.nodes[grid_qubit]["patch_type"] == "ancilla":
            labels[grid_qubit] = "a"
            G.nodes[grid_qubit]["color"] = "cyan"

    def animate(i):
        moment_ops = ops[i]
        G = lay.layout_graph
        for edge in G.edges:
            G.edges[edge]["color"] = "black"
        for op in moment_ops:
            if op.gate in cirq.GateFamily(lsp.Merge):
                if G.nodes[op.qubits[0]]["patch_type"] == "factory":
                    G.nodes[op.qubits[0]]["color"] = "gray"
                merging_qubits = list(op.qubits)
                for i in range(0, len(merging_qubits) - 1):
                    G.edges[(merging_qubits[i], merging_qubits[i + 1])]["color"] = "yellow"
            if op.gate in cirq.GateFamily(lsp.Cultivate):
                G.nodes[op.qubits[0]]["color"] = "green"
        node_colors = []
        edge_colors = []
        for node in list(G.nodes):
            node_colors.append(G.nodes[node]["color"])
        for edge in list(G.edges):
            edge_colors.append(G.edges[edge]["color"])
        nx.draw(
            G,
            pos=pos_dict,
            with_labels=True,
            font_size=12,
            labels=labels,
            node_color=node_colors,
            edge_color=edge_colors,
        )

    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(
        fig, animate, frames=len(ops), interval=1000, blit=False, repeat=False
    )
    plt.show()


def animate_layout_moment(
    G: nx.Graph, moment_paths: list[list[str]], column_layout: Layout
):  # pragma: no cover
    """
    Not sure if this visualization works anymore, hard to get the moment_paths
    """
    moment_paths_flat = list(chain.from_iterable(moment_paths))
    diction = {}
    for qubit in column_layout.qubits:
        diction[qubit["name"]] = (qubit["qubit"].col, -qubit["qubit"].row)
        if qubit["type"] == "qubit":
            G.nodes[qubit["name"]]["color"] = "red"
        elif qubit["type"] == "factory":
            G.nodes[qubit["name"]]["color"] = "green"
        else:
            G.nodes[qubit["name"]]["color"] = "cyan"
    for edge in G.edges:
        G.edges[edge]["color"] = "black"
    for edge in G.edges:
        G.edges[edge]["color"] = "black"

    def animate(i):
        if i != 0:
            if G.nodes[moment_paths_flat[i]]["color"] == "yellow":
                G.nodes[moment_paths_flat[i]]["color"] = "magenta"
            else:
                G.nodes[moment_paths_flat[i]]["color"] = "yellow"
            if moment_paths_flat[i - 1][0] == "a" or moment_paths_flat[i][0] == "a":
                G.edges[moment_paths_flat[i - 1], moment_paths_flat[i]]["color"] = "yellow"
        else:
            G.nodes[moment_paths_flat[i]]["color"] = "yellow"
        colors = [node_data["color"] for node_id, node_data in G.nodes(data=True)]
        edge_colors = [G.edges[edge]["color"] for edge in G.edges()]
        nx.draw(G, pos=diction, node_color=colors, edge_color=edge_colors, with_labels=True)

    fig, ax = plt.subplots()
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(moment_paths_flat),
        interval=1000,
        blit=False,
        repeat=False,
    )
    plt.show()


def draw_2d_array_ascii(arr):  # pragma: no cover
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
    rows = len(arr)
    cols = len(arr[0]) if rows > 0 else 0

    if cols == 0:
        print("Empty array, nothing to draw.")
        return

    # Calculate max string length for formatting
    max_len = 0
    for row in arr:
        for s in row:
            max_len = max(max_len, len(str(s)))

    box_width = max_len + 2  # Padding for the box

    for r in range(rows):
        # Top border of boxes
        print("+" + ("-" * box_width + "+") * cols)

        # String content
        for c in range(cols):
            s = str(arr[r][c])
            if s[0] == "q":
                color = RED
            elif s[0] == "f":
                color = GREEN
            else:
                color = BLUE
            print(color + s.center(box_width) + RESET + "|", end="")
        print("")

        # Bottom border of boxes and connecting lines
        if r < rows - 1:
            print("+" + ("-" * box_width + "+") * cols)

    # Final bottom border
    print("+" + ("-" * box_width + "+") * cols)


class C:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"


def boxed_header(title, width=40):
    pad = width - len(title) - 2
    left = pad // 2
    right = pad - left
    return f"{'=' * left} {title} {'=' * right}"


def hr(width=40):  # pragma: no cover
    return "=" * width


def make_pretty(obj) -> str:  # pragma: no cover
    """
    Pulling out the pretty functionality from the ResourceEstimator class to avoid doubling resource calls
    """
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)
