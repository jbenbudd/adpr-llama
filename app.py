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

def create_interactive_visualization(sequence: str, predicted_sites: List[str]):
    """Create realistic interactive 3D protein structure visualization using Plotly"""
    
    if len(sequence) > 1000:  # Reasonable limit
        # Return a simple text message for very long sequences
        fig = go.Figure()
        fig.add_annotation(
            text=f"Sequence too long for visualization<br>Length: {len(sequence)} residues (max: 1000)<br>Sites: {', '.join(predicted_sites) if predicted_sites else 'None'}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16), align="center"
        )
        fig.update_layout(
            title="Sequence Too Long",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=800, height=400
        )
        return fig
    
    # Parse site positions
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)  # Convert to 0-based
    
    # Amino acid properties for coloring and structure prediction
    hydrophobic = set('AILMFPWYV')
    positive = set('RHK')
    negative = set('DE')
    polar = set('STYNQC')
    special = set('GP')
    
    def get_aa_color(aa):
        """Get color based on amino acid properties"""
        if aa in hydrophobic:
            return 'orange'
        elif aa in positive:
            return 'blue'
        elif aa in negative:
            return 'red'
        elif aa in polar:
            return 'green'
        elif aa in special:
            return 'purple'
        else:
            return 'gray'
    
    def predict_secondary_structure(sequence):
        """Simple secondary structure prediction based on amino acid propensities"""
        structure = []
        for i, aa in enumerate(sequence):
            # Simple heuristic: helix-forming residues tend to form helices
            helix_formers = set('AEHILMRTV')
            sheet_formers = set('FIVWY')
            
            if aa in helix_formers:
                structure.append('H')  # Helix
            elif aa in sheet_formers and i > 2 and i < len(sequence) - 3:
                structure.append('S')  # Sheet
            else:
                structure.append('C')  # Coil
        return structure
    
    # Generate realistic 3D coordinates
    secondary_structure = predict_secondary_structure(sequence)
    
    x_coords = []
    y_coords = []
    z_coords = []
    colors = []
    
    # Starting position
    x, y, z = 0.0, 0.0, 0.0
    phi, psi, omega = 0.0, 0.0, 0.0  # Backbone dihedral angles
    
    # Realistic bond lengths and angles
    ca_ca_distance = 3.8  # Average C-alpha to C-alpha distance
    
    for i, (aa, ss) in enumerate(zip(sequence, secondary_structure)):
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        colors.append(get_aa_color(aa))
        
        # Calculate next position based on secondary structure
        if ss == 'H':  # Alpha helix
            phi_angle = -60  # degrees
            psi_angle = -45
            # Helical geometry
            phi += np.radians(100)  # ~3.6 residues per turn
            x += ca_ca_distance * np.cos(phi) * 0.6
            y += ca_ca_distance * np.sin(phi) * 0.6
            z += 1.5  # Rise per residue in helix
            
        elif ss == 'S':  # Beta sheet
            phi_angle = -120  # degrees
            psi_angle = 120
            # Extended conformation
            direction = (-1) ** (i // 10)  # Alternate direction every 10 residues
            x += ca_ca_distance * 0.9 * direction
            y += ca_ca_distance * 0.3 * np.sin(i * 0.5)
            z += 0.5
            
        else:  # Random coil
            # More random movement
            phi += np.random.uniform(-np.pi/3, np.pi/3)
            psi += np.random.uniform(-np.pi/4, np.pi/4)
            
            x += ca_ca_distance * np.cos(phi) * np.random.uniform(0.7, 1.0)
            y += ca_ca_distance * np.sin(phi) * np.random.uniform(0.7, 1.0)
            z += np.random.uniform(0.5, 2.0)
    
    # Create the plot
    fig = go.Figure()
    
    # Add protein backbone as a ribbon/tube
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines+markers',
        line=dict(color='lightblue', width=12),
        marker=dict(
            size=6, 
            color=colors,
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        name='Protein Backbone',
        hovertemplate='<b>Position %{text}</b><br>Residue: %{customdata[0]}<br>Type: %{customdata[1]}<br>Secondary Structure: %{customdata[2]}<extra></extra>',
        text=[i+1 for i in range(len(sequence))],
        customdata=[[aa, get_aa_color(aa).title(), 
                    'Alpha Helix' if ss == 'H' else 'Beta Sheet' if ss == 'S' else 'Random Coil'] 
                   for aa, ss in zip(sequence, secondary_structure)]
    ))
    
    # Add side chains as smaller spheres
    side_chain_x = []
    side_chain_y = []
    side_chain_z = []
    side_chain_colors = []
    side_chain_text = []
    
    for i, (aa, x, y, z) in enumerate(zip(sequence, x_coords, y_coords, z_coords)):
        # Add side chain at slight offset
        offset_angle = np.random.uniform(0, 2*np.pi)
        offset_distance = 1.5
        
        sc_x = x + offset_distance * np.cos(offset_angle)
        sc_y = y + offset_distance * np.sin(offset_angle)
        sc_z = z + np.random.uniform(-0.5, 0.5)
        
        side_chain_x.append(sc_x)
        side_chain_y.append(sc_y)
        side_chain_z.append(sc_z)
        side_chain_colors.append(get_aa_color(aa))
        side_chain_text.append(f"{aa}{i+1}")
    
    fig.add_trace(go.Scatter3d(
        x=side_chain_x, y=side_chain_y, z=side_chain_z,
        mode='markers',
        marker=dict(
            size=3,
            color=side_chain_colors,
            opacity=0.6
        ),
        name='Side Chains',
        showlegend=False,
        hovertemplate='<b>Side Chain</b><br>%{text}<extra></extra>',
        text=side_chain_text
    ))
    
    # Add PTM sites as large highlighted spheres with glow effect
    if site_positions:
        ptm_x = [x_coords[i] for i in site_positions]
        ptm_y = [y_coords[i] for i in site_positions]
        ptm_z = [z_coords[i] for i in site_positions]
        ptm_labels = [f"{sequence[i]}{i+1}" for i in site_positions]
        ptm_residues = [sequence[i] for i in site_positions]
        
        # Main PTM markers
        fig.add_trace(go.Scatter3d(
            x=ptm_x, y=ptm_y, z=ptm_z,
            mode='markers+text',
            marker=dict(
                size=20,
                color='red',
                opacity=1.0,
                line=dict(color='darkred', width=3),
                symbol='circle'
            ),
            text=ptm_labels,
            textposition="top center",
            textfont=dict(color='red', size=14, family="Arial Black"),
            name='ADP-ribosylation Sites',
            hovertemplate='<b>🔴 ADP-ribosylation Site</b><br>Position: %{text}<br>Residue: %{customdata}<extra></extra>',
            customdata=ptm_residues
        ))
        
        # Add glow effect for PTM sites
        fig.add_trace(go.Scatter3d(
            x=ptm_x, y=ptm_y, z=ptm_z,
            mode='markers',
            marker=dict(
                size=35,
                color='red',
                opacity=0.3,
                line=dict(color='red', width=0)
            ),
            name='PTM Glow',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text=f'Realistic 3D Protein Structure Visualization<br><sub>{len(sequence)} residues, {len(predicted_sites)} ADP-ribosylation sites predicted</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            aspectmode='cube',
            bgcolor='rgba(240,240,240,0.1)',
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="lightgray", 
                showbackground=True,
                zerolinecolor="lightgray",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="lightgray",
                showbackground=True,
                zerolinecolor="lightgray",
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

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
                    output_structure = gr.Plot(label="Interactive 3D Structure")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_analysis, output_sequence, output_structure]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 