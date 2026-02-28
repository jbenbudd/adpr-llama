#!/usr/bin/env python3
"""
Simple adpr-llama Gradio app for ADP-ribosylation site prediction
Uses PEFT adapter model with Zero GPU support
"""

import re
from typing import List, Optional

import gradio as gr
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import requests
import spaces
import plotly.graph_objects as go

# Model configuration
MODEL_REPO = "jbenbudd/ADPrLlama"
MODEL_REVISION = "bb35aa92145ba2b6eba78542ae65e7bc7bdb06bc"  # Set to a specific commit hash like "abc123def456" if needed
CHUNK_SIZE = 21
PAD_CHAR = "-"

print(f"Loading model from {MODEL_REPO}" + (f" at revision {MODEL_REVISION}" if MODEL_REVISION else ""))

# Global variables for model caching
model = None
tokenizer = None

@spaces.GPU
def generate_prediction(prompt: str) -> str:
    """Generate prediction using the model on GPU"""
    global model, tokenizer
    
    try:
        # Load model inside GPU context if not already loaded
        if model is None:
            print("Loading model...")
            model = AutoPeftModelForCausalLM.from_pretrained(
                MODEL_REPO,
                revision=MODEL_REVISION,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_REPO, 
                revision=MODEL_REVISION,
                use_fast=True
            )
            print("Model loaded successfully!")
        
        print(f"Generating prediction for prompt length: {len(prompt)}")
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        print(f"Generated response: {response}")
        return response
        
    except Exception as e:
        print(f"Error in generate_prediction: {e}")
        raise gr.Error(f"Model prediction failed: {str(e)}")

def clean_sequence(sequence: str) -> str:
    """Remove non-amino acid characters and convert to uppercase"""
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())

def chunk_sequence(sequence: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split sequence into chunks of specified size, padding if necessary"""
    chunks = []
    for i in range(0, len(sequence), chunk_size):
        chunk = sequence[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk.ljust(chunk_size, PAD_CHAR)
        chunks.append(chunk)
    return chunks

def parse_sites(text: str) -> List[str]:
    """Extract site predictions from model output"""
    match = re.search(r"Sites=<([^>]*)>", text)
    if not match:
        return []
    sites_str = match.group(1).strip()
    if not sites_str or sites_str.lower() == 'none':
        return []
    return [site.strip() for site in sites_str.split(',') if site.strip()]

def remap_sites(sites: List[str], chunk_index: int, original_length: int, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Remap site positions from chunk-relative to sequence-relative"""
    remapped = []
    chunk_start = chunk_index * chunk_size
    
    for site in sites:
        if not site:
            continue
        
        # Extract residue letter and position
        match = re.match(r'([A-Z])(\d+)', site)
        if not match:
            continue
        
        residue, pos_str = match.groups()
        pos = int(pos_str)
        
        # Convert to 0-based, add chunk offset, convert back to 1-based
        global_pos = chunk_start + (pos - 1) + 1
        
        # Skip if position is beyond original sequence (padding)
        if global_pos <= original_length:
            remapped.append(f"{residue}{global_pos}")
    
    return remapped

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESMFOLD_MAX_LENGTH = 400

def get_pdb_from_esmfold(sequence: str) -> Optional[str]:
    """Predict 3D structure via the ESMFold public API and return the PDB string."""
    try:
        response = requests.post(
            ESMFOLD_API_URL,
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=120,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"ESMFold API error: {e}")
        return None


def create_3dmol_html(pdb_string: str, sequence: str, predicted_sites: List[str]) -> str:
    """Build an HTML document embedding a 3Dmol.js viewer with highlighted PTM sites."""

    site_positions = []
    site_labels = []
    for site in predicted_sites:
        m = re.match(r"([A-Z])(\d+)", site)
        if m:
            site_positions.append(int(m.group(2)))
            site_labels.append(site)

    resi_js_array = "[" + ",".join(str(p) for p in site_positions) + "]"
    labels_js_array = "[" + ",".join(f'"{lbl}"' for lbl in site_labels) + "]"

    pdb_escaped = pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; font-family: Arial, sans-serif; }}
  #container {{ width: 100%; height: 600px; position: relative; }}
  #legend {{
    position: absolute; bottom: 12px; left: 12px; z-index: 10;
    background: rgba(26,26,46,0.85); color: #ccc; padding: 10px 14px;
    border-radius: 8px; font-size: 12px; line-height: 1.6;
    border: 1px solid rgba(255,255,255,0.1);
  }}
  #legend b {{ color: #fff; }}
  .dot {{
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 5px; vertical-align: middle;
  }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
</head>
<body>
<div id="container"></div>
<div id="legend">
  <b>Legend</b><br>
  <span class="dot" style="background:#ef4444;"></span> ADP-ribosylation site<br>
  <span class="dot" style="background:linear-gradient(90deg,#3b82f6,#22c55e,#eab308,#ef4444);width:40px;border-radius:3px;height:8px;"></span> Backbone (N&rarr;C spectrum)
</div>
<script>
$(function() {{
  var pdb = `{pdb_escaped}`;
  var ptmResidues = {resi_js_array};
  var ptmLabels  = {labels_js_array};

  var viewer = $3Dmol.createViewer($("#container"), {{
    backgroundColor: "0x1a1a2e"
  }});
  viewer.addModel(pdb, "pdb");

  // Cartoon backbone coloured by residue index (spectrum)
  viewer.setStyle({{}}, {{
    cartoon: {{ color: "spectrum", opacity: 0.9 }}
  }});

  // PTM sites: add stick + sphere highlight
  if (ptmResidues.length > 0) {{
    viewer.setStyle({{ resi: ptmResidues }}, {{
      cartoon: {{ color: "spectrum", opacity: 0.9 }},
      stick:   {{ radius: 0.25, color: "#ef4444" }},
      sphere:  {{ radius: 0.7,  color: "#ef4444", opacity: 0.85 }}
    }});

    // Labels for each PTM site
    for (var i = 0; i < ptmResidues.length; i++) {{
      var atoms = viewer.getModel(0).selectedAtoms({{ resi: ptmResidues[i], atom: "CA" }});
      if (atoms.length > 0) {{
        viewer.addLabel(ptmLabels[i], {{
          position: atoms[0],
          backgroundColor: "rgba(239,68,68,0.8)",
          fontColor: "white",
          fontSize: 13,
          borderRadius: 4,
          padding: 3,
          showBackground: true
        }});
      }}
    }}
  }}

  viewer.zoomTo();
  viewer.render();
  viewer.zoom(0.85, 800);
}});
</script>
</body>
</html>"""

    return f'<iframe style="width:100%;height:620px;border:none;border-radius:8px;" srcdoc=\'{html}\'></iframe>'


def _structure_unavailable_html(sequence: str, predicted_sites: List[str], reason: str) -> str:
    """Return fallback HTML when ESMFold is not available."""
    sites_str = ", ".join(predicted_sites) if predicted_sites else "None"
    return f"""
    <div style="padding:32px;text-align:center;background:#f8f9fa;border:1px solid #dee2e6;
                border-radius:8px;color:#212529;min-height:300px;display:flex;
                flex-direction:column;justify-content:center;align-items:center;">
        <h3 style="margin:0 0 12px 0;">3D Structure Unavailable</h3>
        <p style="margin:0 0 8px 0;">{reason}</p>
        <p style="margin:0;"><strong>Predicted sites:</strong> {sites_str}</p>
    </div>
    """


def create_sequence_plot(sequence: str, predicted_sites: List[str]):
    """Create a 2D sequence visualization using Plotly"""
    
    # Parse site positions
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)  # Convert to 0-based
    
    # Create sequence grid
    residues_per_row = min(50, len(sequence))
    rows_needed = (len(sequence) + residues_per_row - 1) // residues_per_row
    
    # Prepare data for heatmap
    grid_data = []
    annotations = []
    
    for row in range(rows_needed):
        row_data = []
        for col in range(residues_per_row):
            seq_idx = row * residues_per_row + col
            if seq_idx < len(sequence):
                # 1 for PTM sites, 0 for normal residues
                value = 1 if seq_idx in site_positions else 0
                row_data.append(value)
                
                # Add annotation for amino acid letter
                annotations.append(
                    dict(
                        x=col, y=rows_needed - row - 1,
                        text=sequence[seq_idx],
                        showarrow=False,
                        font=dict(color='white' if seq_idx in site_positions else 'black', size=10)
                    )
                )
                
                # Add position number for PTM sites
                if seq_idx in site_positions:
                    annotations.append(
                        dict(
                            x=col, y=rows_needed - row - 1 - 0.3,
                            text=str(seq_idx + 1),
                            showarrow=False,
                            font=dict(color='white', size=8)
                        )
                    )
            else:
                row_data.append(-1)  # Empty cell
        grid_data.append(row_data)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid_data,
        colorscale=[[0, 'lightblue'], [0.5, 'lightgray'], [1, 'red']],
        showscale=False,
        hovertemplate='Position: %{customdata}<br>Residue: %{text}<extra></extra>',
    ))
    
    # Add annotations
    for ann in annotations:
        fig.add_annotation(**ann)
    
    fig.update_layout(
        title=f'Sequence Layout - {len(sequence)} residues',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=200 + (rows_needed * 30),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def predict_adpr_sites(user_sequence: str):
    """Main prediction function"""
    if not user_sequence.strip():
        return "Please enter a sequence", None, None, None
    
    # Clean and prepare sequence
    clean_seq = clean_sequence(user_sequence)
    if not clean_seq:
        return "Invalid sequence. Please enter amino acid letters only.", None, None, None
    
    original_length = len(clean_seq)
    chunks = chunk_sequence(clean_seq)
    
    all_sites = []
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        
        # Create the exact prompt format from your notebook
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
[Predict the ADP Ribosylation sites given the peptide sequence]
Seq=<{chunk}>

### Response:
"""
        
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk}")
        
        # Generate prediction
        response = generate_prediction(prompt)
        
        # Parse and remap sites
        sites = parse_sites(response)
        remapped_sites = remap_sites(sites, i, original_length)
        all_sites.extend(remapped_sites)
    
    # Format results
    if all_sites:
        sites_text = ", ".join(all_sites)
        highlighted = f"<p><strong>Predicted ADP-ribosylation sites:</strong> {sites_text}</p>"
        highlighted += f"<p><strong>Sequence:</strong> {clean_seq}</p>"
    else:
        highlighted = f"<p><strong>No ADP-ribosylation sites predicted</strong></p>"
        highlighted += f"<p><strong>Sequence:</strong> {clean_seq}</p>"
    
    # Analysis summary
    analysis = f"""
    <div style="padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; color: #212529;">
        <h4 style="color: #495057; margin-top: 0;">Sequence Analysis</h4>
        <p style="color: #212529;"><strong>Original length:</strong> {original_length} residues</p>
        <p style="color: #212529;"><strong>Chunks processed:</strong> {len(chunks)}</p>
        <p style="color: #212529;"><strong>Sites found:</strong> {len(all_sites)}</p>
    </div>
    """
    
    # Create interactive visualizations
    sequence_plot = create_sequence_plot(clean_seq, all_sites)

    if len(clean_seq) > ESMFOLD_MAX_LENGTH:
        structure_html = _structure_unavailable_html(
            clean_seq, all_sites,
            f"Sequence too long for structure prediction ({len(clean_seq)} residues, max {ESMFOLD_MAX_LENGTH}).",
        )
    else:
        pdb_string = get_pdb_from_esmfold(clean_seq)
        if pdb_string:
            structure_html = create_3dmol_html(pdb_string, clean_seq, all_sites)
        else:
            structure_html = _structure_unavailable_html(
                clean_seq, all_sites,
                "The ESMFold structure prediction service is currently unavailable. Please try again later.",
            )

    return highlighted, analysis, sequence_plot, structure_html

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Glass(), title="adpr-llama") as demo:
    gr.Markdown("# 🧬 adpr-llama – ADP-ribosylation Site Predictor")
    gr.Markdown("Enter an amino acid sequence to predict ADP-ribosylation sites. Predicted sites are highlighted in red in both sequence and 3D visualizations.")
    

    
    with gr.Row():
        with gr.Column(scale=1):
            sequence_input = gr.Textbox(
                label="Amino Acid Sequence",
                placeholder="Enter your amino acid sequence (e.g., MASVTIGPLCYRHKNQDEFWQ)",
                lines=3
            )
            
            predict_btn = gr.Button("🧬 Predict Sites", variant="primary")
            
            gr.Examples(
                examples=[
                    ["SLLSKVSQGKRKRGCSHPGGS"]
                ],
                inputs=sequence_input,
                label="Example"
            )
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Predicted Sites"):
                    output_sites = gr.HTML(label="Results")
                with gr.TabItem("Analysis"):
                    output_analysis = gr.HTML(label="Analysis")
                with gr.TabItem("Sequence Layout"):
                    output_sequence = gr.Plot(label="2D Sequence Visualization")
                with gr.TabItem("3D Structure"):
                    output_structure = gr.HTML(label="Interactive 3D Structure")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_analysis, output_sequence, output_structure]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 