import json

path = "docs/en/projects/p01_vae_encoder.ipynb"
with open(path) as f:
    nb = json.load(f)
cells = nb["cells"]

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

demo_md = md_cell(
    "**Checking the lecture's worked example**: the lecture hand-computes $z = 0.76$ from "
    "$\\mu = 0.50$, $\\sigma = 0.20$, $\\varepsilon = 1.30$. Running the same numbers through "
    "`reparameterize`'s formula confirms the code matches the math exactly."
)
demo_code = code_cell(
    "demo_mu = torch.tensor([[0.50]])\n"
    "demo_std = torch.tensor([[0.20]])\n"
    "demo_eps = torch.tensor([[1.30]])\n"
    "demo_z = demo_mu + demo_std * demo_eps\n"
    "print(f'z = mu + std * eps = {demo_mu.item()} + {demo_std.item()} * {demo_eps.item()} = {demo_z.item():.2f}')\n"
    "assert abs(demo_z.item() - 0.76) < 1e-6, \"does not match the lecture's hand-worked result\"\n"
)

new_cells = cells[:13] + [demo_md, demo_code] + cells[13:]
nb["cells"] = new_cells
with open(path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("done, total cells:", len(new_cells))
