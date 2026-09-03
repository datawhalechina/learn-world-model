import json, sys

def split_notebook(path, narrative_1, narrative_2, narrative_3):
    with open(path) as f:
        nb = json.load(f)
    cells = nb["cells"]
    old_code = "".join(cells[8]["source"])

    dec_idx = old_code.index("class Decoder")
    vae_idx = old_code.index("class VAE")

    part1 = old_code[:dec_idx].rstrip() + "\n"
    part2 = old_code[dec_idx:vae_idx].rstrip() + "\n"
    part3 = old_code[vae_idx:]

    def code_cell(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

    def md_cell(src):
        return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

    new_cells = (
        cells[:7]
        + [md_cell(narrative_1), code_cell(part1)]
        + [md_cell(narrative_2), code_cell(part2)]
        + [md_cell(narrative_3), code_cell(part3)]
        + cells[9:]
    )
    nb["cells"] = new_cells
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

if __name__ == "__main__":
    split_notebook(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
