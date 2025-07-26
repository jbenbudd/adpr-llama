#!/usr/bin/env python3
"""
Simple adpr-llama Gradio app for ADP-ribosylation site prediction
Uses PEFT adapter model with Zero GPU support
"""

import re
from typing import List, Tuple
import io
import base64

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import numpy as np
import spaces
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
                device_map="auto",  # Zero GPU will handle device placement
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

def generate_pdb_structure(sequence: str, predicted_sites: List[str]):
    """Generate a PDB structure from amino acid sequence"""
    
    # Parse site positions
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)  # Convert to 0-based
    
    def predict_secondary_structure(sequence):
        """Simple secondary structure prediction"""
        structure = []
        for i, aa in enumerate(sequence):
            helix_formers = set('AEHILMRTV')
            sheet_formers = set('FIVWY')
            
            if aa in helix_formers:
                structure.append('H')  # Helix
            elif aa in sheet_formers and i > 2 and i < len(sequence) - 3:
                structure.append('S')  # Sheet
            else:
                structure.append('C')  # Coil
        return structure
    
    secondary_structure = predict_secondary_structure(sequence)
    
    # Generate realistic 3D coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    
    # Starting position
    x, y, z = 0.0, 0.0, 0.0
    phi = 0.0
    
    # Realistic bond lengths
    ca_ca_distance = 3.8
    
    for i, (aa, ss) in enumerate(zip(sequence, secondary_structure)):
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        
        # Calculate next position based on secondary structure
        if ss == 'H':  # Alpha helix
            phi += np.radians(100)  # ~3.6 residues per turn
            x += ca_ca_distance * np.cos(phi) * 0.6
            y += ca_ca_distance * np.sin(phi) * 0.6
            z += 1.5  # Rise per residue in helix
        elif ss == 'S':  # Beta sheet
            direction = (-1) ** (i // 10)
            x += ca_ca_distance * 0.9 * direction
            y += ca_ca_distance * 0.3 * np.sin(i * 0.5)
            z += 0.5
        else:  # Random coil
            phi += np.random.uniform(-np.pi/3, np.pi/3)
            x += ca_ca_distance * np.cos(phi) * np.random.uniform(0.7, 1.0)
            y += ca_ca_distance * np.sin(phi) * np.random.uniform(0.7, 1.0)
            z += np.random.uniform(0.5, 2.0)
    
    # Generate PDB content
    pdb_lines = []
    pdb_lines.append("HEADER    PREDICTED PROTEIN STRUCTURE")
    pdb_lines.append("TITLE     ADP-RIBOSYLATION SITE PREDICTION")
    pdb_lines.append("MODEL        1")
    
    for i, (aa, x, y, z, ss) in enumerate(zip(sequence, x_coords, y_coords, z_coords, secondary_structure)):
        # ATOM record format: ATOM + serial + name + alt + resName + chainID + resSeq + iCode + x + y + z + occupancy + tempFactor + element
        atom_line = f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        pdb_lines.append(atom_line)
    
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")
    pdb_content = "\n".join(pdb_lines)
    
    return pdb_content, site_positions

def create_interactive_visualization(sequence: str, predicted_sites: List[str]):
    """Create realistic molecular visualization using 3Dmol.js"""
    
    if len(sequence) > 1000:
        return f"""
        <div style="text-align: center; padding: 50px; font-size: 16px; background: white;">
            <h3>Sequence Too Long for Visualization</h3>
            <p>Length: {len(sequence)} residues (max: 1000)</p>
            <p>Predicted Sites: {', '.join(predicted_sites) if predicted_sites else 'None'}</p>
        </div>
        """
    
    # Generate PDB structure
    pdb_content, site_positions = generate_pdb_structure(sequence, predicted_sites)
    
    # Escape the PDB content for JavaScript
    pdb_escaped = pdb_content.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    
    # Create selection strings for PTM sites
    ptm_selection = ""
    ptm_labels = ""
    if site_positions:
        ptm_residues = [str(pos + 1) for pos in site_positions]  # Convert to 1-based
        ptm_selection = " or ".join([f"resi:{res}" for res in ptm_residues])
        
        # Generate label commands
        label_commands = []
        for pos in site_positions:
            res_num = pos + 1
            residue = sequence[pos]
            label_commands.append(f"""
                viewer.addLabel('{residue}{res_num}', {{resi:{res_num}}}, 
                              {{fontSize: 12, fontColor: 'red', backgroundColor: 'white', 
                                backgroundOpacity: 0.8, borderColor: 'red', borderWidth: 1}});
            """)
        ptm_labels = "".join(label_commands)
    
    # Create 3Dmol.js visualization HTML
    html_content = f"""
    <div style="width: 100%; font-family: Arial, sans-serif;">
        <div style="text-align: center; margin-bottom: 15px; background: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h3 style="color: #333; margin: 0 0 10px 0;">🧬 Realistic Protein Structure Visualization</h3>
            <p style="color: #666; margin: 0;">
                <strong>Sequence:</strong> {len(sequence)} residues | 
                <strong>ADP-ribosylation Sites:</strong> {len(predicted_sites)} predicted
                {f' | <strong>Sites:</strong> {", ".join(predicted_sites)}' if predicted_sites else ''}
            </p>
        </div>
        
        <div style="text-align: center; margin-bottom: 10px;">
            <button onclick="resetView()" style="margin: 0 5px; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Reset View</button>
            <button onclick="toggleStyle()" style="margin: 0 5px; padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Toggle Style</button>
            <button onclick="toggleSites()" style="margin: 0 5px; padding: 8px 16px; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer;">Highlight Sites</button>
        </div>
        
        <div id="molviewer" style="width: 800px; height: 600px; margin: 0 auto; border: 2px solid #ddd; border-radius: 8px; background: white;"></div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
        <script>
            let viewer;
            let currentStyle = 'cartoon';
            let sitesVisible = true;
            
            // Initialize when DOM is ready
            document.addEventListener('DOMContentLoaded', function() {{
                initViewer();
            }});
            
            function initViewer() {{
                const element = document.getElementById('molviewer');
                if (!element) {{
                    console.error('Viewer element not found');
                    return;
                }}
                
                viewer = $3Dmol.createViewer(element, {{
                    defaultcolors: $3Dmol.rasmolElementColors
                }});
                
                const pdbData = `{pdb_escaped}`;
                
                try {{
                    viewer.addModel(pdbData, "pdb");
                    setCartoonStyle();
                    {ptm_labels}
                    {f'highlightPTMSites();' if ptm_selection else ''}
                    viewer.zoomTo();
                    viewer.render();
                }} catch (error) {{
                    console.error('Error initializing viewer:', error);
                    element.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;">Error loading molecular structure</div>';
                }}
            }}
            
            function setCartoonStyle() {{
                viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                currentStyle = 'cartoon';
            }}
            
            function setSphereStyle() {{
                viewer.setStyle({{}}, {{sphere: {{color: 'spectrum', radius: 0.8}}}});
                currentStyle = 'sphere';
            }}
            
            function setStickStyle() {{
                viewer.setStyle({{}}, {{stick: {{color: 'spectrum', radius: 0.3}}}});
                currentStyle = 'stick';
            }}
            
            function highlightPTMSites() {{
                if (sitesVisible && "{ptm_selection}") {{
                    viewer.addStyle({{{ptm_selection}}}, {{
                        sphere: {{color: 'red', radius: 1.2}},
                        stick: {{color: 'red', radius: 0.4}}
                    }});
                }}
            }}
            
            function toggleStyle() {{
                if (currentStyle === 'cartoon') {{
                    setSphereStyle();
                }} else if (currentStyle === 'sphere') {{
                    setStickStyle();
                }} else {{
                    setCartoonStyle();
                }}
                
                {f'highlightPTMSites();' if ptm_selection else ''}
                viewer.render();
            }}
            
            function toggleSites() {{
                sitesVisible = !sitesVisible;
                viewer.removeAllLabels();
                
                if (sitesVisible) {{
                    {ptm_labels}
                    {f'highlightPTMSites();' if ptm_selection else ''}
                }} else {{
                    if (currentStyle === 'cartoon') {{
                        setCartoonStyle();
                    }} else if (currentStyle === 'sphere') {{
                        setSphereStyle();
                    }} else {{
                        setStickStyle();
                    }}
                }}
                viewer.render();
            }}
            
            function resetView() {{
                viewer.zoomTo();
                viewer.render();
            }}
            
            // Initialize if DOM is already loaded
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', initViewer);
            }} else {{
                setTimeout(initViewer, 100);
            }}
        </script>
    </div>
    """
    
    return html_content

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
    structure_plot = create_interactive_visualization(clean_seq, all_sites)
    
    return highlighted, analysis, sequence_plot, structure_plot

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
                    output_structure = gr.HTML(label="Realistic 3D Structure")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_analysis, output_sequence, output_structure]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 