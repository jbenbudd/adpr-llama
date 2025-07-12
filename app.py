import re
from typing import List, Tuple

import gradio as gr
from huggingface_hub import InferenceClient

MODEL_REPO = "jbenbudd/ADPrLlama"
CHUNK_SIZE = 21  # model context length for sequences
PAD_CHAR = "-"  # character used for right-padding short sequences

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def clean_sequence(seq: str) -> str:
    """Return sequence as uppercase letters only (A-Z)."""
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def chunk_sequence(seq: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split sequence into fixed-length chunks, padding the last chunk on the right."""
    chunks = []
    for i in range(0, len(seq), chunk_size):
        chunk = seq[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + PAD_CHAR * (chunk_size - len(chunk))
        chunks.append(chunk)
    return chunks


def parse_sites(raw_output: str) -> List[Tuple[str, int]]:
    """Extract residues and positions from model output `Sites=<A2,I7,...>`.

    Returns list of tuples (residue_letter, position_int).
    """
    match = re.search(r"Sites=<([^>]+)>", raw_output)
    if not match:
        return []
    site_str = match.group(1)
    sites = []
    for item in site_str.split(","):
        item = item.strip()
        if len(item) < 2:
            continue
        residue = item[0]
        try:
            pos = int(item[1:])
            sites.append((residue, pos))
        except ValueError:
            continue
    return sites


def remap_sites(chunks: List[str], chunk_sites: List[List[Tuple[str, int]]]) -> List[Tuple[str, int]]:
    """Convert per-chunk site predictions to positions in the original, unchunked sequence."""
    global_sites = []
    for chunk_index, (chunk, sites) in enumerate(zip(chunks, chunk_sites)):
        start_pos = chunk_index * CHUNK_SIZE  # 0-based index in original sequence
        real_length = len(chunk.rstrip(PAD_CHAR))
        for residue, local_pos in sites:
            if local_pos <= real_length:  # ignore predictions falling in padded region
                global_pos = start_pos + local_pos  # convert to 1-based global index later
                global_sites.append((residue, global_pos))
    return global_sites


def format_sites(sites: List[Tuple[str, int]]) -> str:
    """Return string representation `Sites=<A2,I7,...>` sorted by position."""
    if not sites:
        return "Sites=<None>"
    # sort by position
    sites_sorted = sorted(sites, key=lambda x: x[1])
    inner = ",".join(f"{res}{pos}" for res, pos in sites_sorted)
    return f"Sites=<{inner}>"


def highlight_sequence(seq: str, sites: List[Tuple[str, int]]) -> str:
    """Return HTML string highlighting predicted sites in red bold."""
    site_positions = {pos for _, pos in sites}  # 1-based positions
    parts = []
    for i, aa in enumerate(seq, start=1):
        if i in site_positions:
            parts.append(f"<span style='color:red;font-weight:bold'>{aa}</span>")
        else:
            parts.append(aa)
    return "<div style='font-family:monospace;font-size:16px'>" + "".join(parts) + "</div>"

# Add mapping for residue names
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "-": "GLY",  # pad as glycine (ignored)
}


def generate_pdb(seq: str) -> str:
    """Generate a pseudo-structure PDB string with one Cα per residue on a helix."""
    lines = []
    import math

    radius = 10.0
    rise_per_res = 1.5
    angle_per_res = math.radians(100)  # ~3.6 residues per turn

    serial = 1
    for i, aa in enumerate(seq, start=1):
        if aa == PAD_CHAR:
            continue  # skip pads in structure
        theta = angle_per_res * (i - 1)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = rise_per_res * (i - 1)
        resname = ONE_TO_THREE.get(aa, "GLY")
        line = (
            f"ATOM  {serial:5d}  CA  {resname} A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)
        serial += 1
    lines.append("END")
    return "\n".join(lines)


def create_ngl_html(pdb_str: str, site_positions: List[int]) -> str:
    """Return HTML/JS snippet embedding NGL viewer with colored PTM sites."""
    import html
    pdb_escaped = html.escape(pdb_str)
    sites_js_array = ",".join(str(p) for p in site_positions)
    # Use template string
    return f"""
    <div id='nglviewer' style='width:100%; height:400px;'></div>
    <script src='https://cdn.jsdelivr.net/npm/ngl@2.0.0-dev.35/dist/ngl.js'></script>
    <script>
      const stage = new NGL.Stage('nglviewer');
      const pdbText = `{pdb_escaped}`;
      const blob = new Blob([pdbText], {{type: 'text/plain'}});
      stage.loadFile(blob, {{ ext: 'pdb' }}).then(function(comp) {{
          comp.addRepresentation('cartoon', {{
              color: function(atom) {{
                  const sites = [{sites_js_array}];
                  if (sites.includes(atom.resno)) return 'red';
                  return 'lightgrey';
              }}
          }});
          stage.autoView();
      }});
    </script>
    """

# ---------------------------------------------------------
# Inference client setup
# ---------------------------------------------------------

client = InferenceClient(model=MODEL_REPO)

def predict_adpr_sites(user_sequence: str):
    """Gradio interface callback to predict ADP-ribosylation sites."""
    if not user_sequence:
        return "", "Please enter a protein sequence."

    sequence = clean_sequence(user_sequence)
    if not sequence:
        return "", "Invalid input: sequence must contain alphabetic characters (A-Z)."

    chunks = chunk_sequence(sequence)

    chunk_sites: List[List[Tuple[str, int]]] = []
    for chunk in chunks:
        prompt = f"Seq=<{chunk}>"
        # Query model. Use a small max_new_tokens: output is short.
        generation = client.text_generation(prompt, max_new_tokens=15, temperature=0.0)
        # Example model output: "Sites=<A2,I7,F19>"
        sites = parse_sites(generation)
        chunk_sites.append(sites)

    global_sites = remap_sites(chunks, chunk_sites)
    sites_str = format_sites(global_sites)
    highlighted_html = highlight_sequence(sequence, global_sites)
    # 3D visualization
    pdb_str = generate_pdb(sequence)
    site_positions = [pos for _, pos in global_sites]
    ngl_html = create_ngl_html(pdb_str, site_positions)

    return sites_str, highlighted_html, ngl_html


DESCRIPTION = (
    "Enter an amino acid sequence (any length). The model will automatically split the sequence "
    "into 21-length segments, predict potential ADP-ribosylation (ADPr) sites, and aggregate the "
    "results back onto the original sequence. Predicted sites are highlighted in the visualization "
    "below."
)

logo_md = "## adpr-llama : ADP-ribosylation Site Predictor"

with gr.Blocks(title="adpr-llama") as demo:
    gr.Markdown(logo_md)
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        seq_input = gr.Textbox(
            label="Amino Acid Sequence",
            placeholder="MASVTIGPLCYRHKNQDEFWQ",
            lines=4,
            elem_id="sequence-input",
        )

    with gr.Row():
        result_box = gr.Textbox(label="Predicted ADPr Sites", interactive=False)

    with gr.Row():
        html_view = gr.HTML(label="Sequence Visualization")
    with gr.Row():
        ngl_view = gr.HTML(label="3D Viewer")

    predict_btn = gr.Button("Predict")

    predict_btn.click(predict_adpr_sites, inputs=[seq_input], outputs=[result_box, html_view, ngl_view])

    # Allow pressing Enter in the textbox to trigger prediction
    seq_input.submit(predict_adpr_sites, inputs=[seq_input], outputs=[result_box, html_view, ngl_view])

if __name__ == "__main__":
    demo.launch() 