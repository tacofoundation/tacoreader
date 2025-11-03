"""HTML representation for TacoDataset (Jupyter notebook display)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tacoreader.loader import TacoDataset


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
    # Add 1 to include root level (level 0)
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
    # Add 1 to include root level (level 0)
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
    # Add 1 to include root level
    total_levels = len(hierarchy) + 1

    # Build hierarchy list
    items = []

    # Root level
    root_type = dataset.pit_schema.root["type"]
    root_n = dataset.pit_schema.root["n"]
    items.append(f"<li><strong>Level 0:</strong> {root_type} × {root_n}</li>")

    # Child levels
    for depth_str in sorted(hierarchy.keys(), key=int):
        depth = int(depth_str)
        patterns = hierarchy[depth_str]

        for pattern_idx, pattern in enumerate(patterns):
            types = pattern["type"]
            ids = pattern["id"]
            n = pattern.get("n", "?")

            types_str = f"[{', '.join(types)}]"

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
                <strong>Level {depth}:</strong> {types_str} × {n} each
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

        # Show ALL fields, no truncation
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


def render_graph(dataset: "TacoDataset") -> str:
    """
    Render PIT schema graph as SVG (right side).

    Shows:
    - ROOT (TACO) at top
    - Sample homogeneity: [Sample0, ..., SampleN]
    - Children structure below

    Args:
        dataset: TacoDataset instance

    Returns:
        SVG string for graph visualization
    """
    pit_schema = dataset.pit_schema
    hierarchy = pit_schema.hierarchy
    root = pit_schema.root

    # Graph configuration
    NODE_RADIUS = 12
    LEVEL_HEIGHT = 60
    NODE_SPACING = 45
    START_X = 110
    START_Y = 30

    nodes = []
    edges = []
    texts = []  # For "..." text labels

    # ===== ROOT NODE (TACO - Pink) =====
    root_node = {
        "id": "root",
        "type": "TACO",
        "level": -1,
        "x": START_X,
        "y": START_Y,
        "label": "TACO",
        "color": "#FFB6C1",  # Pink
    }
    nodes.append(root_node)

    # ===== LEVEL 0 - Show homogeneity pattern =====
    # [Sample_left, ..., Sample_right] to represent all samples share structure
    level0_y = START_Y + LEVEL_HEIGHT
    n_samples = root["n"]

    # Left sample
    sample_left = {
        "id": "sample_0",
        "type": root["type"],
        "level": 0,
        "x": START_X - NODE_SPACING,
        "y": level0_y,
        "label": "Sample 0",
        "color": get_node_color(root["type"]),
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

    # Dots in the middle (text, not a node)
    texts.append({"x": START_X, "y": level0_y + 5, "text": "..."})

    # Right sample
    sample_right = {
        "id": f"sample_{n_samples-1}",
        "type": root["type"],
        "level": 0,
        "x": START_X + NODE_SPACING,
        "y": level0_y,
        "label": f"Sample {n_samples-1}",
        "color": get_node_color(root["type"]),
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

    # ===== LEVEL 1+ - Show children of first sample =====
    # We'll show children below the LEFT sample to keep it compact
    parent_node = sample_left
    current_y = level0_y

    # Only show first level of children to keep graph compact
    if hierarchy and "1" in hierarchy:
        patterns = hierarchy["1"]
        if patterns:
            pattern = patterns[0]
            types = pattern["type"]
            ids = pattern["id"]

            # Show max 3 children to avoid overcrowding
            max_children = min(3, len(types))

            current_y = level0_y + LEVEL_HEIGHT

            # Calculate spacing
            total_width = (max_children - 1) * NODE_SPACING
            start_x = parent_node["x"] - (total_width / 2)

            for i in range(max_children):
                child_type = types[i]
                child_id = ids[i]

                child_node = {
                    "id": f"child_{i}",
                    "type": child_type,
                    "level": 1,
                    "x": start_x + (i * NODE_SPACING),
                    "y": current_y,
                    "label": child_id[:8],  # Truncate long IDs
                    "color": get_node_color(child_type),
                }
                nodes.append(child_node)

                # Edge from parent to child
                edges.append(
                    {
                        "from_x": parent_node["x"],
                        "from_y": parent_node["y"],
                        "to_x": child_node["x"],
                        "to_y": child_node["y"],
                    }
                )

            # If there are more children, show "..."
            if len(types) > max_children:
                texts.append(
                    {
                        "x": start_x + (max_children * NODE_SPACING),
                        "y": current_y + 5,
                        "text": f"... +{len(types) - max_children}",
                    }
                )

    # Calculate SVG dimensions
    svg_width = 220
    svg_height = current_y + 40

    # ===== Render SVG =====

    # Edges
    svg_edges = []
    for edge in edges:
        dx = edge["to_x"] - edge["from_x"]
        dy = edge["to_y"] - edge["from_y"]
        length = (dx**2 + dy**2) ** 0.5

        if length > 0:
            # Shorten line by radius amount
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

    # Nodes
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

    # Text labels (for "...")
    svg_texts = []
    for text in texts:
        svg_texts.append(
            f"""
        <text x="{text['x']}" y="{text['y']}" 
              text-anchor="middle" 
              font-size="14" 
              font-weight="bold"
              fill="#666">{text['text']}</text>
        """
        )

    svg = f"""
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

    return svg


def get_node_color(node_type: str) -> str:
    """
    Get color for node based on type.

    Args:
        node_type: "TACO", "FOLDER", or "FILE"

    Returns:
        Hex color string
    """
    if node_type == "TACO":
        return "#FFB6C1"  # Pink (ROOT/TACO)
    elif node_type == "FOLDER":
        return "#90EE90"  # Light green
    elif node_type == "FILE":
        return "#DDA0DD"  # Plum (light purple)
    else:
        return "#FFB6C1"  # Pink (default for unknown)


def get_css() -> str:
    """
    Get CSS styles for HTML representation.

    Returns:
        CSS string
    """
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

    /* Header */
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

    /* Body layout */
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
        flex: 0 0 220px;
        align-self: flex-start;
    }

    /* Sections */
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

    /* Lists */
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

    /* Sub-details (IDs) */
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

    /* Attributes table */
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

    /* Graph */
    .taco-graph-svg {
        display: block;
        margin: 0 auto;
    }

    /* Responsive */
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
