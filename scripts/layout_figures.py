# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cirq
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))
import resource_estimation as res

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


DEFAULT_STR2NUM = {
    "None": 0,
    "T Factory": 1,
    "S Factory": 2,
    "Data Qubit": 3,
    "Ancilla Patch": 4,
    "Distillation": 5,
}


def layout_to_array(layout, str2num=None):
    """
    Convert a layout object's `layout_graph` into a 2D numpy array.

    Parameters
    ----------
    layout : object
        Must have `layout_graph` with nodes exposing `.row` and `.col`.
    str2num : dict, optional
        Mapping from string type to integer code.

    Returns
    -------
    np.ndarray
        Array A such that A[row, col] contains the numeric code for that node.
    """
    if str2num is None:
        str2num = DEFAULT_STR2NUM

    G = layout.layout_graph

    max_row = max(node.row for node in G.nodes)
    max_col = max(node.col for node in G.nodes)

    A = np.full((max_row + 1, max_col + 1), str2num["None"], dtype=int)
    key_map = {"data": "Data Qubit", "ancilla": "Ancilla Patch", "t": "T Factory", "s": "S Factory", "block": "Distillation"}
    for node in G.nodes:
        node_dict = G.nodes[node]
        key = node_dict["ftype"] if "ftype" in node_dict else node_dict["patch_type"]
        key = key_map[key]
        A[node.row, node.col] = str2num[key]

    return A


def make_layout_colormap(
    str2num=None,
    none_color="gray",
    base_cmap_name="tab10",
    category_colors=None,
):
    """
    Build a discrete colormap and normalization for categorical plotting.

    Parameters
    ----------
    str2num : dict
        Mapping from category name to integer.
    none_color : color
        Fallback color for "None" if category_colors is not provided.
    base_cmap_name : str
        Matplotlib colormap name to use if category_colors is not provided.
    category_colors : dict or list, optional
        If dict, should map category names to colors, e.g.
            {"None": "#d9d9d9", "t": "#1f77b4", ...}
        If list, should be ordered by integer value:
            [color_for_0, color_for_1, color_for_2, ...]
    """
    if str2num is None:
        str2num = DEFAULT_STR2NUM

    n = len(str2num)

    if category_colors is not None:
        if isinstance(category_colors, dict):
            colors = [None] * n
            for name, idx in str2num.items():
                if name not in category_colors:
                    raise ValueError(f"Missing color for category {name!r}")
                colors[idx] = category_colors[name]
        else:
            colors = list(category_colors)
            if len(colors) != n:
                raise ValueError(f"category_colors must have length {n}, got {len(colors)}")
    else:
        base_cmap = plt.get_cmap(base_cmap_name)
        colors = [none_color] + [base_cmap(i) for i in range(1, n)]

    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(0, n + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def make_legend_patches(str2num=None, cmap=None):
    """
    Create legend patches for the categorical colormap.
    """
    if str2num is None:
        str2num = DEFAULT_STR2NUM
    if cmap is None:
        cmap, _ = make_layout_colormap(str2num=str2num)

    num2str = {v: k for k, v in str2num.items()}

    return [mpatches.Patch(color=cmap(i), label=num2str[i]) for i in range(len(str2num))]


def pad_array_to_shape(A, target_shape, fill_value=0, align="center"):
    """
    Pad a 2D array to a target shape.

    Parameters
    ----------
    A : np.ndarray
        Input array.
    target_shape : tuple[int, int]
        Desired output shape (rows, cols).
    fill_value : int, optional
        Padding value.
    align : str, optional
        One of:
        - "center"
        - "top_left"
        - "top_right"
        - "bottom_left"
        - "bottom_right"

    Returns
    -------
    np.ndarray
        Padded array.
    """
    target_rows, target_cols = target_shape
    rows, cols = A.shape

    if rows > target_rows or cols > target_cols:
        raise ValueError("target_shape must be at least as large as A.shape")

    row_diff = target_rows - rows
    col_diff = target_cols - cols

    if align == "center":
        pad_top = row_diff // 2
        pad_bottom = row_diff - pad_top
        pad_left = col_diff // 2
        pad_right = col_diff - pad_left
    elif align == "top_left":
        pad_top, pad_bottom = 0, row_diff
        pad_left, pad_right = 0, col_diff
    elif align == "top_right":
        pad_top, pad_bottom = 0, row_diff
        pad_left, pad_right = col_diff, 0
    elif align == "bottom_left":
        pad_top, pad_bottom = row_diff, 0
        pad_left, pad_right = 0, col_diff
    elif align == "bottom_right":
        pad_top, pad_bottom = row_diff, 0
        pad_left, pad_right = col_diff, 0
    else:
        raise ValueError(f"Unknown align={align!r}")

    return np.pad(
        A,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=fill_value,
    )


def normalize_transpose_arg(transpose, n):
    """
    Allow transpose to be either a single bool or a list of bools.
    """
    if isinstance(transpose, bool):
        return [transpose] * n

    if len(transpose) != n:
        raise ValueError("transpose must be a bool or a sequence matching the number of layouts")

    return list(transpose)


def add_pixel_grid(ax, shape, linewidth=0.5, color="black"):
    """
    Add borders around each displayed pixel/cell.
    """
    nrows, ncols = shape

    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=color, linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)


def style_axes(ax, border_linewidth=2):
    """
    Remove major ticks and add a black border around the axes.
    """
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(border_linewidth)


def plot_layout(
    layout,
    ax=None,
    str2num=None,
    none_color="gray",
    base_cmap_name="tab10",
    transpose=True,
    title=None,
    show_legend=True,
    legend_kwargs=None,
    show_pixel_grid=True,
    grid_linewidth=0.5,
    grid_color="black",
    border_linewidth=2,
    category_colors=None,
):
    """
    Plot a single layout.

    Parameters
    ----------
    layout : object
        Layout object with `layout_graph`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, create a new figure.
    str2num : dict, optional
        Mapping of category names to integer codes.
    none_color : str, optional
        Color for the "None" category.
    base_cmap_name : str, optional
        Base categorical colormap.
    transpose : bool, optional
        Whether to plot A.T instead of A.
    title : str, optional
        Title for the subplot.
    show_legend : bool, optional
        Whether to show a legend.
    legend_kwargs : dict, optional
        Extra keyword args for legend placement/styling.
    show_pixel_grid : bool, optional
        Whether to draw borders around each pixel.
    grid_linewidth : float, optional
        Pixel grid line width.
    grid_color : str, optional
        Pixel grid line color.
    border_linewidth : float, optional
        Outer axes border width.

    Returns
    -------
    fig, ax, im
    """
    if str2num is None:
        str2num = DEFAULT_STR2NUM

    A = layout_to_array(layout, str2num=str2num)
    plot_array = A.T if transpose else A

    cmap, norm = make_layout_colormap(
        str2num=str2num,
        none_color=none_color,
        base_cmap_name=base_cmap_name,
        category_colors=category_colors,
    )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(plot_array, cmap=cmap, norm=norm, interpolation="auto")
    ax.set_aspect("equal")

    style_axes(ax, border_linewidth=border_linewidth)

    if show_pixel_grid:
        add_pixel_grid(ax, plot_array.shape, linewidth=grid_linewidth, color=grid_color)

    if title is not None:
        ax.set_title(title)

    if show_legend:
        patches = make_legend_patches(str2num=str2num, cmap=cmap)
        default_legend_kwargs = {"bbox_to_anchor": (1.05, 1), "loc": "upper left"}
        if legend_kwargs is not None:
            default_legend_kwargs.update(legend_kwargs)
        ax.legend(handles=patches, **default_legend_kwargs)

    if created_fig:
        fig.tight_layout()

    return fig, ax, im


def plot_layouts(
    layouts,
    titles=None,
    str2num=None,
    none_color="gray",
    base_cmap_name="tab10",
    transpose=True,
    ncols=None,
    figsize=None,
    shared_legend=True,
    legend_kwargs=None,
    enforce_same_shape=True,
    align="center",
    show_pixel_grid=True,
    grid_linewidth=0.5,
    grid_color="black",
    border_linewidth=2,
    category_colors=None,
):
    """
    Plot multiple layouts in a single figure.

    Parameters
    ----------
    layouts : sequence
        Sequence of layout objects.
    titles : sequence[str], optional
        Titles for each subplot.
    str2num : dict, optional
        Mapping of category names to integer codes.
    none_color : str, optional
        Color for the "None" category.
    base_cmap_name : str, optional
        Base categorical colormap.
    transpose : bool or sequence[bool], optional
        Either one value for all subplots, or one bool per layout.
    ncols : int, optional
        Number of subplot columns.
    figsize : tuple, optional
        Figure size.
    shared_legend : bool, optional
        Whether to draw one figure-level legend.
    legend_kwargs : dict, optional
        Extra keyword args for legend placement/styling.
    enforce_same_shape : bool, optional
        Pad all layouts to the same shape for fair visual comparison.
    align : str, optional
        Alignment mode used during padding.
    show_pixel_grid : bool, optional
        Whether to draw borders around each pixel.
    grid_linewidth : float, optional
        Pixel grid line width.
    grid_color : str, optional
        Pixel grid line color.
    border_linewidth : float, optional
        Outer axes border width.

    Returns
    -------
    fig, axes
    """
    if str2num is None:
        str2num = DEFAULT_STR2NUM

    n = len(layouts)
    if n == 0:
        raise ValueError("layouts must contain at least one layout")

    if titles is not None and len(titles) != n:
        raise ValueError("titles must match the number of layouts")

    transpose_list = normalize_transpose_arg(transpose, n)

    arrays = [layout_to_array(layout, str2num=str2num) for layout in layouts]

    if enforce_same_shape:
        max_rows = max(A.shape[0] for A in arrays)
        max_cols = max(A.shape[1] for A in arrays)
        target_shape = (max_rows, max_cols)

        arrays = [
            pad_array_to_shape(
                A,
                target_shape=target_shape,
                fill_value=str2num["None"],
                align=align,
            )
            for A in arrays
        ]

    if ncols is None:
        ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    cmap, norm = make_layout_colormap(
        str2num=str2num,
        none_color=none_color,
        base_cmap_name=base_cmap_name,
        category_colors=category_colors,
    )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, (A, do_transpose) in enumerate(zip(arrays, transpose_list)):
        ax = axes[i]
        plot_array = A.T if do_transpose else A

        ax.imshow(plot_array, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_aspect("equal")

        style_axes(ax, border_linewidth=border_linewidth)

        if show_pixel_grid:
            add_pixel_grid(ax, plot_array.shape, linewidth=grid_linewidth, color=grid_color)

        if titles is not None:
            ax.set_title(titles[i])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if shared_legend:
        patches = make_legend_patches(str2num=str2num, cmap=cmap)
        default_legend_kwargs = {"loc": "center left", "bbox_to_anchor": (0.85, 0.5)}
        if legend_kwargs is not None:
            default_legend_kwargs.update(legend_kwargs)
        fig.legend(handles=patches, **default_legend_kwargs)

        fig.subplots_adjust(right=0.82)
    else:
        fig.tight_layout()

    return fig, axes


if __name__ == "__main__":
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2,
            "axes.linewidth": 1.5,
            "grid.alpha": 0.4,
            "figure.dpi": 150,
        }
    )
    circuit = cirq.read_json("circuits/20-features-90sparse.json")

    embedded_layout = res.layout.Embedded(input_circuit=circuit)
    column_layout = res.layout.Column(input_circuit=circuit)
    sandwich1 = res.layout.FactorySandwich(
        input_circuit=circuit, num_s_factories=10, num_t_factories=20
    )
    sandwich2 = res.layout.FactorySandwich(
        input_circuit=circuit, num_s_factories=15, num_t_factories=5
    )
    movement1 = res.layout.MovementLayout(input_circuit=circuit, num_t_factories=10)
    movement2 = res.layout.MovementLayout(input_circuit=circuit, num_t_factories=20)
    distillery = res.layout.Distillery(input_circuit=circuit, num_s_factories=7, num_t_factories=5)
    layouts = [
            column_layout, 
            sandwich1, sandwich2, 
            embedded_layout, 
            movement1, 
            distillery
        ]

    category_colors = {
        "None": "#ffffff",
        "T Factory": "#4ddef1",
        "S Factory": "#fcc084",
        "Data Qubit": "#a4f0c2",
        "Ancilla Patch": "#f5bad6",
        "Distillation": "#E6E6FA",
    }

    fig, axes = plot_layouts(
        layouts,
        titles=[
            "Column",
            "Sandwich s=10, t=20",
            "Sandwich s=15, t=5",
            "Embedded",
            "Movement t=10",
            "Movement t=20",
        ],
        transpose=[False, False, False, False, True, True],
        category_colors=category_colors,
        shared_legend=True,
        ncols=3,
        figsize=(14, 7),
        grid_linewidth=1,
        grid_color="#c9c9c9",
        border_linewidth=2,
        enforce_same_shape=True,
    )
    filename = "notebooks/figures_for_paper/layouts.pdf"
    # plt.savefig(filename, bbox_inches="tight")
    # print(f"Saved to {filename}")
    plt.show()
