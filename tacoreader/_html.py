"""HTML representation for TacoDataset (Jupyter notebook display)."""

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from tacoreader.loader import TacoDataset


class NodeDict(TypedDict):
    """Type for graph node dictionaries."""

    id: str
    type: str
    level: int
    x: float
    y: float
    label: str
    color: str


class EdgeDict(TypedDict):
    """Type for graph edge dictionaries."""

    from_x: float
    from_y: float
    to_x: float
    to_y: float


class TextDict(TypedDict, total=False):
    """Type for graph text dictionaries."""

    x: float
    y: float
    text: str
    size: str


# Graph configuration constants
NODE_RADIUS = 12
LEVEL_HEIGHT = 55
NODE_SPACING_H = 40
NODE_SPACING_V = 50
START_X = 110
START_Y = 30
MAX_CHILDREN_FULL = 4
LEFT_MARGIN = NODE_RADIUS + 10


def build_html_repr(dataset: "TacoDataset") -> str:
    """
    Build complete HTML representation for TacoDataset.

    Args:
        dataset: TacoDataset instance

    Returns:
        HTML string for Jupyter display
    """
    header = render_header(dataset)
    sections = render_sections(dataset)
    graph = render_graph(dataset)
    css = get_css()

    html = f"""
    <div class="taco-dataset-wrap">
        {header}
        <div class="taco-body">
            <div class="taco-sections">
                {sections}
            </div>
            <div class="taco-graph">
                {graph}
            </div>
        </div>
        <style>{css}</style>
    </div>
    """
    return html


def render_header(dataset: "TacoDataset") -> str:
    """
    Render header with dataset title and name.

    Args:
        dataset: TacoDataset instance

    Returns:
        HTML string for header
    """
    title = dataset.title if dataset.title else dataset.id
    n_samples = dataset.pit_schema.root["n"]
    total_levels = dataset.pit_schema.max_depth() + 1

    return f"""
    <div class="taco-header">
        <div class="taco-obj-type">&lt;TacoDataset&gt;</div>
        <div class="taco-obj-name">{title}</div>
        <div class="taco-obj-dims">({n_samples} samples, {total_levels} levels)</div>
    </div>
    """


def render_sections(dataset: "TacoDataset") -> str:
    """
    Render left side sections (Dimensions, Hierarchy, Metadata, Fields).

    Args:
        dataset: TacoDataset instance

    Returns:
        HTML string for sections
    """
    dimensions_section = render_dimensions_section(dataset)
    hierarchy_section = render_hierarchy_section(dataset)
    metadata_section = render_metadata_section(dataset)
    fields_section = render_fields_section(dataset)

    return f"""
    {dimensions_section}
    {hierarchy_section}
    {metadata_section}
    {fields_section}
    """


def render_dimensions_section(dataset: "TacoDataset") -> str:
    """Render Dimensions section (overview)."""
    n_samples = dataset.pit_schema.root["n"]
    total_levels = dataset.pit_schema.max_depth() + 1
    format_type = dataset._format.upper()

    return f"""
    <details class="taco-section" open>
        <summary>
            <span class="taco-section-arrow">▼</span>
            <span class="taco-section-title">Dimensions:</span>
            <span class="taco-section-count">(3)</span>
        </summary>
        <div class="taco-section-content">
            <ul class="taco-dim-list">
                <li><strong>samples:</strong> {n_samples}</li>
                <li><strong>depth:</strong> {total_levels}</li>
                <li><strong>format:</strong> {format_type}</li>
            </ul>
        </div>
    </details>
    """


def render_hierarchy_section(dataset: "TacoDataset") -> str:
    """Render Hierarchy section with level details."""
    hierarchy = dataset.pit_schema.hierarchy
    total_levels = len(hierarchy) + 1

    items = []

    # Root level
    root_type = dataset.pit_schema.root["type"]
    root_n = dataset.pit_schema.root["n"]
    items.append(f"<li><strong>Level 0:</strong> {root_type} x {root_n}</li>")

    # Child levels - group patterns by level
    for depth_str in sorted(hierarchy.keys(), key=int):
        depth = int(depth_str)
        patterns = hierarchy[depth_str]

        if not patterns:
            continue

        # Take first pattern as representative (all should be identical due to PIT)
        pattern = patterns[0]
        types = pattern["type"]
        ids = pattern["id"]
        n = pattern.get("n", "?")
        num_patterns = len(patterns)

        types_str = f"[{', '.join(types)}]"

        # Show pattern count if multiple
        pattern_info = f" x {num_patterns} positions" if num_patterns > 1 else ""

        # Sub-details for IDs
        ids_html = f"""
        <details class="taco-sub-details">
            <summary>▶ Show IDs ({len(ids)})</summary>
            <ul class="taco-id-list">
                {''.join(f'<li><span class="taco-id">{id_}</span> ({typ})</li>' for id_, typ in zip(ids, types, strict=False))}
            </ul>
        </details>
        """

        items.append(
            f"""
        <li>
            <strong>Level {depth}:</strong> {types_str}{pattern_info} x {n} each
            {ids_html}
        </li>
        """
        )

    hierarchy_html = "".join(items)

    return f"""
    <details class="taco-section">
        <summary>
            <span class="taco-section-arrow">▶</span>
            <span class="taco-section-title">Hierarchy:</span>
            <span class="taco-section-count">({total_levels} levels)</span>
        </summary>
        <div class="taco-section-content">
            <ul class="taco-hierarchy-list">
                {hierarchy_html}
            </ul>
        </div>
    </details>
    """


def render_metadata_section(dataset: "TacoDataset") -> str:
    """Render Metadata/Attributes section."""
    attrs = [
        ("id", dataset.id),
        ("version", dataset.version),
        (
            "description",
            (
                dataset.description[:100] + "..."
                if len(dataset.description) > 100
                else dataset.description
            ),
        ),
        ("tasks", ", ".join(dataset.tasks)),
        ("licenses", ", ".join(dataset.licenses)),
    ]

    # Add spatial extent if available
    if "spatial" in dataset.extent:
        spatial = dataset.extent["spatial"]
        attrs.append(
            (
                "spatial",
                f"[{spatial[0]:.1f}, {spatial[1]:.1f}, {spatial[2]:.1f}, {spatial[3]:.1f}]",
            )
        )

    # Add temporal extent if available
    if dataset.extent.get("temporal"):
        temporal = dataset.extent["temporal"]
        attrs.append(("temporal", f"{temporal[0]} → {temporal[1]}"))

    rows = "".join(
        f"<tr><td class='taco-attr-name'>{name}</td><td class='taco-attr-value'>{value}</td></tr>"
        for name, value in attrs
    )

    return f"""
    <details class="taco-section">
        <summary>
            <span class="taco-section-arrow">▶</span>
            <span class="taco-section-title">Attributes:</span>
            <span class="taco-section-count">({len(attrs)})</span>
        </summary>
        <div class="taco-section-content">
            <table class="taco-attrs-table">
                {rows}
            </table>
        </div>
    </details>
    """


def render_fields_section(dataset: "TacoDataset") -> str:
    """Render Fields section (fields by level)."""
    field_schema = dataset.field_schema
    total_fields = sum(len(fields) for fields in field_schema.values())

    rows = []
    for level_key in sorted(field_schema.keys()):
        fields = field_schema[level_key]
        field_names = [f[0] for f in fields]
        field_count = len(field_names)

        field_list = ", ".join(field_names)

        rows.append(
            f"<li><strong>{level_key}:</strong> ({field_count}) {field_list}</li>"
        )

    fields_html = "".join(rows)

    return f"""
    <details class="taco-section">
        <summary>
            <span class="taco-section-arrow">▶</span>
            <span class="taco-section-title">Fields by level:</span>
            <span class="taco-section-count">({total_fields} total)</span>
        </summary>
        <div class="taco-section-content">
            <ul class="taco-fields-list">
                {fields_html}
            </ul>
        </div>
    </details>
    """


def _create_root_node() -> NodeDict:
    """Create the root TACO node."""
    return {
        "id": "root",
        "type": "TACO",
        "level": -1,
        "x": START_X,
        "y": START_Y,
        "label": "TACO",
        "color": "#FFB6C1",
    }


def _create_sample_level(
    root_node: NodeDict,
    root_type: str,
    n_samples: int,
    nodes: list[NodeDict],
    edges: list[EdgeDict],
    texts: list[TextDict],
) -> NodeDict:
    """
    Create level 0 showing sample homogeneity pattern.

    Returns the left sample node that will be expanded.
    """
    level0_y = START_Y + LEVEL_HEIGHT

    # Left sample (will be expanded)
    sample_left: NodeDict = {
        "id": "sample_0",
        "type": root_type,
        "level": 0,
        "x": START_X - NODE_SPACING_H,
        "y": level0_y,
        "label": "Sample 0",
        "color": get_node_color(root_type),
    }
    nodes.append(sample_left)
    edges.append(
        {
            "from_x": root_node["x"],
            "from_y": root_node["y"],
            "to_x": sample_left["x"],
            "to_y": sample_left["y"],
        }
    )

    # Dots in the middle
    texts.append({"x": START_X, "y": level0_y + 5, "text": "..."})

    # Right sample
    sample_right: NodeDict = {
        "id": f"sample_{n_samples-1}",
        "type": root_type,
        "level": 0,
        "x": START_X + NODE_SPACING_H,
        "y": level0_y,
        "label": f"Sample {n_samples-1}",
        "color": get_node_color(root_type),
    }
    nodes.append(sample_right)
    edges.append(
        {
            "from_x": root_node["x"],
            "from_y": root_node["y"],
            "to_x": sample_right["x"],
            "to_y": sample_right["y"],
        }
    )

    return sample_left


def _determine_children_to_show(num_children: int) -> tuple[list[int], bool, int]:
    """
    Determine which children indices to show based on count.

    Returns:
        (indices_to_show, show_ellipsis, hidden_count)
    """
    if num_children <= MAX_CHILDREN_FULL:
        return list(range(num_children)), False, 0
    else:
        # Show first 2, ellipsis, last 1
        return [0, 1, num_children - 1], True, num_children - 3


def _create_child_nodes(
    parent_node: NodeDict,
    types: list[str],
    ids: list[str],
    current_depth: int,
    current_y: float,
    indices_to_show: list[int],
    show_ellipsis: bool,
    hidden_count: int,
    nodes: list[NodeDict],
    edges: list[EdgeDict],
    texts: list[TextDict],
) -> NodeDict | None:
    """
    Create child nodes for a level.

    Returns the first FOLDER child for expansion, or None.
    """
    num_visible = len(indices_to_show) + (1 if show_ellipsis else 0)
    total_width = (num_visible - 1) * NODE_SPACING_H
    start_x = parent_node["x"] - (total_width / 2)

    expanded_child: NodeDict | None = None
    current_x = start_x

    for idx_pos, i in enumerate(indices_to_show):
        child_type = types[i]
        child_id = ids[i]

        child_node: NodeDict = {
            "id": f"child_d{current_depth}_i{i}",
            "type": child_type,
            "level": current_depth,
            "x": current_x,
            "y": current_y,
            "label": child_id[:10],
            "color": get_node_color(child_type),
        }
        nodes.append(child_node)

        edges.append(
            {
                "from_x": parent_node["x"],
                "from_y": parent_node["y"],
                "to_x": child_node["x"],
                "to_y": child_node["y"],
            }
        )

        # Mark FIRST FOLDER found for expansion (not just index 0)
        if expanded_child is None and child_type == "FOLDER":
            expanded_child = child_node

        current_x += NODE_SPACING_H

        # Add ellipsis after second node if needed
        if show_ellipsis and idx_pos == 1:
            texts.append(
                {
                    "x": current_x,
                    "y": current_y + 5,
                    "text": f"...(+{hidden_count})",
                    "size": "9",
                }
            )
            current_x += NODE_SPACING_H

    # Show collapsed indicator for sibling folders
    if expanded_child:
        for i in indices_to_show[1:]:
            if types[i] == "FOLDER":
                sibling_node = next(
                    n for n in nodes if n["id"] == f"child_d{current_depth}_i{i}"
                )
                texts.append(
                    {
                        "x": sibling_node["x"],
                        "y": sibling_node["y"] + NODE_SPACING_V // 2,
                        "text": "...",
                        "size": "10",
                    }
                )

    return expanded_child


def _expand_hierarchy(
    parent_node: NodeDict,
    hierarchy: dict,
    start_depth: int,
    start_y: float,
    nodes: list[NodeDict],
    edges: list[EdgeDict],
    texts: list[TextDict],
) -> float:
    """
    Recursively expand first child at each level.

    Returns final Y position.
    """
    current_y = start_y
    current_parent = parent_node
    current_depth = start_depth

    while str(current_depth) in hierarchy:
        patterns = hierarchy[str(current_depth)]
        if not patterns:
            break

        pattern = patterns[0]
        types = pattern["type"]
        ids = pattern["id"]

        current_y += NODE_SPACING_V
        num_children = len(types)

        indices_to_show, show_ellipsis, hidden_count = _determine_children_to_show(
            num_children
        )

        expanded_child = _create_child_nodes(
            current_parent,
            types,
            ids,
            current_depth,
            current_y,
            indices_to_show,
            show_ellipsis,
            hidden_count,
            nodes,
            edges,
            texts,
        )

        if expanded_child:
            current_parent = expanded_child
            current_depth += 1
        else:
            # No more folders to expand
            break

    return current_y


def _adjust_node_positions(
    nodes: list[NodeDict], edges: list[EdgeDict], texts: list[TextDict]
) -> None:
    """Adjust node positions to prevent cutoff on left side."""
    if not nodes:
        return

    min_x = min(node["x"] for node in nodes)

    if min_x < LEFT_MARGIN:
        offset = LEFT_MARGIN - min_x

        for node in nodes:
            node["x"] += offset

        for edge in edges:
            edge["from_x"] += offset
            edge["to_x"] += offset

        for text in texts:
            text["x"] += offset


def _render_svg_elements(
    nodes: list[NodeDict], edges: list[EdgeDict], texts: list[TextDict], final_y: float
) -> str:
    """Render SVG elements (edges, nodes, texts)."""
    svg_edges = []
    for edge in edges:
        dx = edge["to_x"] - edge["from_x"]
        dy = edge["to_y"] - edge["from_y"]
        length = (dx**2 + dy**2) ** 0.5

        if length > 0:
            scale = (length - NODE_RADIUS) / length
            to_x = edge["from_x"] + dx * scale
            to_y = edge["from_y"] + dy * scale

            svg_edges.append(
                f"""
            <line x1="{edge['from_x']}" y1="{edge['from_y']}"
                  x2="{to_x}" y2="{to_y}"
                  stroke="#999" stroke-width="1.5"
                  marker-end="url(#arrowhead)"/>
            """
            )

    svg_nodes = []
    for node in nodes:
        svg_nodes.append(
            f"""
        <circle cx="{node['x']}" cy="{node['y']}" r="{NODE_RADIUS}"
                fill="{node['color']}" stroke="#666" stroke-width="1.5">
            <title>{node['label']} ({node['type']})</title>
        </circle>
        """
        )

    svg_texts = []
    for text in texts:
        size = text.get("size", "12")
        svg_texts.append(
            f"""
        <text x="{text['x']}" y="{text['y']}"
              text-anchor="middle"
              font-size="{size}"
              font-weight="bold"
              fill="#666">{text['text']}</text>
        """
        )

    # Calculate SVG dimensions
    if nodes:
        max_x = max(node["x"] for node in nodes)
        svg_width = int(max_x + NODE_RADIUS + 20)
    else:
        svg_width = 250

    svg_height = final_y + 50

    return f"""
    <svg width="{svg_width}" height="{svg_height}" class="taco-graph-svg">
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="10"
                    refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="#999"/>
            </marker>
        </defs>
        {''.join(svg_edges)}
        {''.join(svg_nodes)}
        {''.join(svg_texts)}
    </svg>
    """


def render_graph(dataset: "TacoDataset") -> str:
    """
    Render PIT schema graph as SVG showing ONE complete sample expanded.

    Shows:
    - ROOT (TACO) at top
    - Sample homogeneity: [Sample0, ..., SampleN]
    - ONE complete sample fully expanded to leaves
    - Smart collapsing: show all if ≤4 children, else show first 2 + "...(+N)" + last
    - Other siblings shown collapsed

    Args:
        dataset: TacoDataset instance

    Returns:
        SVG string for graph visualization
    """
    pit_schema = dataset.pit_schema
    hierarchy = pit_schema.hierarchy
    root = pit_schema.root

    nodes: list[NodeDict] = []
    edges: list[EdgeDict] = []
    texts: list[TextDict] = []

    # Create root node
    root_node = _create_root_node()
    nodes.append(root_node)

    # Create sample level
    sample_left = _create_sample_level(
        root_node, root["type"], root["n"], nodes, edges, texts
    )

    # Expand hierarchy from sample_left
    level0_y = START_Y + LEVEL_HEIGHT
    final_y = _expand_hierarchy(
        sample_left, hierarchy, 1, level0_y, nodes, edges, texts
    )

    # Adjust positions to prevent cutoff
    _adjust_node_positions(nodes, edges, texts)

    # Render SVG
    return _render_svg_elements(nodes, edges, texts, final_y)


def get_node_color(node_type: str) -> str:
    """Get color for node based on type."""
    colors = {
        "TACO": "#FFB6C1",
        "FOLDER": "#90EE90",
        "FILE": "#DDA0DD",
    }
    return colors.get(node_type, "#FFB6C1")


def get_css() -> str:
    """Get CSS styles for HTML representation."""
    return """
    .taco-dataset-wrap {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 12px;
        line-height: 1.5;
        color: var(--jp-ui-font-color0, #000);
        background: var(--jp-layout-color0, #fff);
        border: 1px solid var(--jp-border-color2, #e0e0e0);
        border-radius: 4px;
        padding: 10px;
        margin: 5px 0;
    }

    .taco-header {
        border-bottom: 1px solid var(--jp-border-color2, #e0e0e0);
        padding-bottom: 8px;
        margin-bottom: 10px;
    }

    .taco-obj-type {
        color: var(--jp-ui-font-color2, #666);
        font-family: Monaco, monospace;
        font-size: 11px;
    }

    .taco-obj-name {
        font-size: 14px;
        font-weight: 600;
        color: var(--jp-ui-font-color0, #000);
        margin: 2px 0;
    }

    .taco-obj-dims {
        color: var(--jp-ui-font-color2, #666);
        font-size: 11px;
    }

    .taco-body {
        display: flex;
        flex-direction: row;
        gap: 15px;
    }

    .taco-sections {
        flex: 1;
        min-width: 300px;
    }

    .taco-graph {
        flex: 0 0 260px;
        align-self: flex-start;
    }

    .taco-section {
        margin: 8px 0;
        border-left: 2px solid var(--jp-border-color2, #e0e0e0);
        padding-left: 8px;
    }

    .taco-section summary {
        cursor: pointer;
        user-select: none;
        list-style: none;
        padding: 2px 0;
    }

    .taco-section summary::-webkit-details-marker {
        display: none;
    }

    .taco-section summary:hover {
        background: var(--jp-layout-color2, #f5f5f5);
    }

    .taco-section-arrow {
        display: inline-block;
        width: 12px;
        font-size: 10px;
        color: var(--jp-ui-font-color2, #666);
    }

    .taco-section[open] .taco-section-arrow {
        transform: rotate(0deg);
    }

    .taco-section-title {
        font-weight: 600;
        color: var(--jp-ui-font-color1, #333);
    }

    .taco-section-count {
        color: var(--jp-ui-font-color2, #666);
        font-size: 11px;
        margin-left: 4px;
    }

    .taco-section-content {
        padding: 5px 0 5px 15px;
    }

    .taco-dim-list,
    .taco-hierarchy-list,
    .taco-fields-list {
        list-style: none;
        padding: 0;
        margin: 5px 0;
    }

    .taco-dim-list li,
    .taco-hierarchy-list li,
    .taco-fields-list li {
        padding: 2px 0;
    }

    .taco-id-list {
        list-style: none;
        padding-left: 15px;
        margin: 5px 0;
    }

    .taco-id-list li {
        padding: 1px 0;
        font-size: 11px;
    }

    .taco-id {
        font-family: Monaco, monospace;
        color: var(--jp-mirror-editor-variable-color, #0066cc);
    }

    .taco-sub-details {
        margin: 5px 0;
        padding-left: 10px;
    }

    .taco-sub-details summary {
        cursor: pointer;
        font-size: 11px;
        color: var(--jp-ui-font-color2, #666);
    }

    .taco-sub-details summary:hover {
        background: var(--jp-layout-color2, #f5f5f5);
    }

    .taco-attrs-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 11px;
    }

    .taco-attrs-table td {
        padding: 3px 8px 3px 0;
        vertical-align: top;
    }

    .taco-attr-name {
        font-weight: 600;
        color: var(--jp-ui-font-color1, #333);
        white-space: nowrap;
    }

    .taco-attr-value {
        color: var(--jp-ui-font-color0, #000);
        word-break: break-word;
    }

    .taco-graph-svg {
        display: block;
        margin: 0 auto;
    }

    @media (max-width: 768px) {
        .taco-body {
            flex-direction: column;
        }

        .taco-graph {
            order: -1;
            flex: 0 0 auto;
        }
    }
    """
