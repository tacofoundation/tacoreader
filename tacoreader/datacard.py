import pathlib
from typing import List, Union

from tacoreader.load import load_metadata

try:
    from mdutils.mdutils import MdUtils
except ImportError:
    raise ImportError("Please install the mdutils package with: pip install mdutils")


def read_datacard(
    file: Union[str, pathlib.Path, List[pathlib.Path], List[str]],
    outfile: Union[str, pathlib.Path],
) -> pathlib.Path:

    # Create a JSON file
    taco_object: dict = load_metadata(file)

    # Convert the output file to a pathlib.Path object
    if isinstance(outfile, str):
        outfile = pathlib.Path(outfile)

    # Create the README.md file
    md_file = MdUtils(file_name=outfile)

    # --- YAML Header ---
    md_file.new_line("---")
    md_file.new_line("license:")
    for item in taco_object["licenses"]:
        md_file.new_line(f"  - {item}")
    md_file.new_line("language:")
    md_file.new_line("- en")
    if taco_object["keywords"]:
        md_file.new_line("tags:")
        for tag in taco_object["keywords"]:
            md_file.new_line(f"  - {tag}")
    md_file.new_line(f'pretty_name: {taco_object["id"]}')
    md_file.new_line("---")

    # --- Title and Description ---
    md_file.new_header(level=1, title=taco_object["id"])
    if taco_object["title"]:
        md_file.new_line(f'**{taco_object["title"]}**', bold_italics_code="b")
    md_file.new_paragraph(taco_object["description"])

    # --- Code Snippet ---
    md_file.new_header(level=2, title="🌮 TACO Snippet")
    md_file.new_paragraph("Load this dataset using the `tacoreader` library.")
    md_file.new_line("```python")
    md_file.new_line("import tacoreader")
    md_file.new_line("dataset = tacoreader.load('...')")
    md_file.new_line("```")

    # example in R
    md_file.new_line("\n")
    md_file.new_paragraph("Or in R:")
    md_file.new_line("```r")
    md_file.new_line("library(tacoreader)")
    md_file.new_line("dataset <- tacoreader::load('...')")
    md_file.new_line("```")

    # --- Sensor Information ---
    if taco_object["optical_data"]:
        md_file.new_header(level=2, title="🛰️ Sensor Information")
        md_file.new_paragraph(
            f'The sensor related to the dataset: **{taco_object["optical_data"]["sensor"]}**'
        )

    # --- Task Information ---
    md_file.new_header(level=2, title="🎯 Task")
    md_file.new_paragraph(
        f'The task associated with this dataset: **{taco_object["task"]}**'
    )

    # --- Raw Repository Link ---
    if taco_object["raw_link"]:
        md_file.new_header(level=2, title="📂 Original Data Repository")
        md_file.new_paragraph(
            f'Source location of the raw data:**{md_file.new_inline_link(link=taco_object["raw_link"]["href"])}**'
        )

    # --- Discussion Link ---
    if taco_object["discuss_link"]:
        md_file.new_header(level=2, title="💬 Discussion")
        md_file.new_paragraph(
            f'Insights or clarifications about the dataset: **{md_file.new_inline_link(link=taco_object["discuss_link"]["href"])}**'
        )

    # --- Split Strategy ---
    if taco_object["split_strategy"]:
        md_file.new_header(level=2, title="🔀 Split Strategy")
        md_file.new_paragraph(
            f'How the dataset is divided for training, validation, and testing: **{taco_object["split_strategy"]}**'
        )

    # --- Scientific Publications ---
    if taco_object["scientific"]["publications"]:
        md_file.new_header(level=2, title="📚 Scientific Publications")
        md_file.new_paragraph("Publications that reference or describe the dataset.")
        for idx, pub in enumerate(taco_object["scientific"]["publications"], start=1):
            # Add the publication information
            md_file.new_paragraph(f"### Publication {idx:02d}")
            md_file.new_line("- **DOI**: " + md_file.new_inline_link(link=pub["doi"]))
            md_file.new_line("- **Summary**: " + pub["summary"])
            md_file.new_line("- **BibTeX Citation**:")
            md_file.new_line("```bibtex")
            md_file.new_line(pub["citation"].strip("\n"))
            md_file.new_line("```")
            md_file.new_line("\n")

    # --- Data Providers ---
    if taco_object["providers"]:
        md_file.new_header(level=2, title="🤝 Data Providers")
        md_file.new_paragraph(
            "Organizations or individuals responsible for the dataset."
        )

        # Define table headers
        table_headers = ["**Name**", "**Role**", "**URL**"]
        table_data = [table_headers]

        # Populate table with provider data
        for provider in taco_object["providers"]:
            table_data.append(
                [
                    provider["name"] or "N/A",
                    (
                        ", ".join(provider["roles"])
                        if isinstance(provider["roles"], list)
                        else provider["roles"] or "N/A"
                    ),
                    (
                        md_file.new_inline_link(link=provider["links"][0]["href"])
                        if provider["links"]
                        else "N/A"
                    ),
                ]
            )

        # Flatten the list for Markdown formatting
        flat_table_data = [cell for row in table_data for cell in row]

        # Create the table
        md_file.new_table(
            columns=3, rows=len(table_data), text=flat_table_data, text_align="left"
        )

    # --- Curators ---
    if taco_object["curators"]:
        md_file.new_header(level=2, title="🧑‍🔬 Curators")
        md_file.new_paragraph(
            "Responsible for structuring the dataset in the TACO format."
        )

        # Define table headers
        table_headers = ["**Name**", "**Organization**", "**URL**"]
        table_data = [table_headers]

        # Populate table with curator data
        for curator in taco_object["curators"]:
            table_data.append(
                [
                    curator["name"] or "N/A",
                    curator["organization"] or "N/A",
                    (
                        md_file.new_inline_link(link=curator["links"][0]["href"])
                        if curator["links"]
                        else "N/A"
                    ),
                ]
            )

        # Flatten the list for Markdown formatting
        flat_table_data = [cell for row in table_data for cell in row]
        # Create the table
        md_file.new_table(
            columns=3, rows=len(table_data), text=flat_table_data, text_align="left"
        )

    # --- Labels ---
    if taco_object["labels"]:
        md_file.new_header(level=2, title="🏷️ Labels")
        md_file.new_paragraph(taco_object["labels"]["label_description"])
        table_headers = ["**Name**", "**Category**", "**Description**"]
        table_data = [table_headers]
        for item in taco_object["labels"]["label_classes"]:
            table_data.append(
                [
                    item["name"] or "N/A",
                    (
                        str(item["category"]) if item["category"] is not None else "N/A"
                    ),  # Zero is a valid category
                    item["description"] or "N/A",
                ]
            )

        # Flatten the list for Markdown formatting
        flat_table_data = [cell for row in table_data for cell in row]
        md_file.new_table(
            columns=3, rows=len(table_data), text=flat_table_data, text_align="left"
        )

    # --- Optical Bands ---
    if taco_object["optical_data"]:
        md_file.new_header(level=2, title="🌈 Optical Bands")
        md_file.new_paragraph("Spectral bands related to the sensor.")
        table_headers = [
            "**Name**",
            "**Common Name**",
            "**Description**",
            "**Center Wavelength**",
            "**Full Width Half Max**",
            "**Index**",
        ]
        table_data = [table_headers]

        for item in taco_object["optical_data"]["bands"]:
            table_data.append(
                [
                    item["name"] or "N/A",
                    item["common_name"] or "N/A",
                    item["description"] or "N/A",
                    item["center_wavelength"] or "N/A",
                    item["full_width_half_max"] or "N/A",
                    str(item["index"]) if item["index"] is not None else "N/A",
                ]
            )

        # Flatten the list for Markdown formatting
        flat_table_data = [cell for row in table_data for cell in row]
        md_file.new_table(
            columns=6, rows=len(table_data), text=flat_table_data, text_align="left"
        )

    # Export the data to the output file
    file = md_file.get_md_text().replace("\n\n\n  \n", "").replace("  \n", "\n")
    with open(outfile, "w") as f:
        f.write(file)

    return outfile
