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

def create_interactive_visualization(sequence: str, predicted_sites: List[str]):
    """Create realistic molecular visualization using advanced Plotly techniques"""
    
    if len(sequence) > 1000:
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
    
    # Amino acid properties for realistic coloring
    def get_aa_properties(aa):
        """Get amino acid properties for realistic coloring"""
        hydrophobic = {'A': '#FFA500', 'I': '#FF8C00', 'L': '#FF7F50', 'M': '#FF6347', 
                      'F': '#FF4500', 'P': '#FF1493', 'W': '#8B008B', 'Y': '#9932CC', 'V': '#BA55D3'}
        positive = {'R': '#0000FF', 'H': '#4169E1', 'K': '#6495ED'}
        negative = {'D': '#FF0000', 'E': '#DC143C'}
        polar = {'S': '#32CD32', 'T': '#228B22', 'Y': '#9932CC', 'N': '#90EE90', 'Q': '#98FB98', 'C': '#FFFF00'}
        special = {'G': '#DDA0DD', 'P': '#FF1493'}
        
        if aa in hydrophobic:
            return hydrophobic[aa], 'Hydrophobic'
        elif aa in positive:
            return positive[aa], 'Positive'
        elif aa in negative:
            return negative[aa], 'Negative'
        elif aa in polar:
            return polar[aa], 'Polar'
        elif aa in special:
            return special[aa], 'Special'
        else:
            return '#808080', 'Other'
    
    def predict_secondary_structure(sequence):
        """Enhanced secondary structure prediction"""
        structure = []
        for i, aa in enumerate(sequence):
            # More sophisticated prediction based on amino acid propensities
            helix_formers = set('AEHILMRTV')
            sheet_formers = set('FIVWY')
            loop_formers = set('GSPND')
            
            # Look at local environment
            if i < len(sequence) - 2:
                triplet = sequence[i:i+3]
                if 'PP' in triplet or 'PG' in triplet:
                    structure.append('L')  # Loop/turn
                    continue
            
            if aa in helix_formers and i > 1 and i < len(sequence) - 2:
                structure.append('H')  # Helix
            elif aa in sheet_formers and i > 2 and i < len(sequence) - 3:
                structure.append('S')  # Sheet
            elif aa in loop_formers:
                structure.append('L')  # Loop
            else:
                structure.append('C')  # Coil
        return structure
    
    secondary_structure = predict_secondary_structure(sequence)
    
    # Generate realistic 3D coordinates with improved geometry
    coords = []
    x, y, z = 0.0, 0.0, 0.0
    phi, psi = 0.0, 0.0
    
    # Realistic structural parameters
    ca_ca_distance = 3.8
    helix_pitch = 1.5
    helix_radius = 2.3
    
    for i, (aa, ss) in enumerate(zip(sequence, secondary_structure)):
        coords.append((x, y, z))
        
        if ss == 'H':  # Alpha helix - realistic geometry
            phi += np.radians(100)  # 3.6 residues per turn
            x += helix_radius * np.cos(phi)
            y += helix_radius * np.sin(phi)
            z += helix_pitch
        elif ss == 'S':  # Beta sheet - extended conformation
            direction = (-1) ** (i // 8)  # Alternate direction
            x += ca_ca_distance * 0.95 * direction
            y += ca_ca_distance * 0.2 * np.sin(i * 0.4)
            z += 0.3
        elif ss == 'L':  # Loop - tight turn
            phi += np.random.uniform(-np.pi/2, np.pi/2)
            psi += np.random.uniform(-np.pi/4, np.pi/4)
            x += ca_ca_distance * 0.6 * np.cos(phi)
            y += ca_ca_distance * 0.6 * np.sin(phi)
            z += np.random.uniform(-0.5, 1.0)
        else:  # Random coil
            phi += np.random.uniform(-np.pi/3, np.pi/3)
            x += ca_ca_distance * np.cos(phi) * np.random.uniform(0.8, 1.0)
            y += ca_ca_distance * np.sin(phi) * np.random.uniform(0.8, 1.0)
            z += np.random.uniform(0.5, 1.5)
    
    x_coords, y_coords, z_coords = zip(*coords)
    
    # Create advanced molecular visualization
    fig = go.Figure()
    
    # 1. Protein backbone as ribbon with varying thickness
    colors = [get_aa_properties(aa)[0] for aa in sequence]
    
    # Main backbone ribbon
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines+markers',
        line=dict(
            color=colors,
            width=12,
            colorscale='Viridis'
        ),
        marker=dict(
            size=6,
            color=colors,
            opacity=0.9,
            line=dict(color='white', width=1)
        ),
        name='Backbone',
        hovertemplate='<b>%{customdata[0]}%{customdata[1]}</b><br>' +
                     'Type: %{customdata[2]}<br>' +
                     'Secondary Structure: %{customdata[3]}<br>' +
                     'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>',
        customdata=[[aa, i+1, get_aa_properties(aa)[1], 
                    'α-Helix' if ss == 'H' else 'β-Sheet' if ss == 'S' else 'Loop' if ss == 'L' else 'Random Coil'] 
                   for i, (aa, ss) in enumerate(zip(sequence, secondary_structure))]
    ))
    
    # 2. Secondary structure enhancement
    for i in range(len(x_coords)-1):
        ss = secondary_structure[i]
        
        if ss == 'H':  # Helices - thick ribbons
            fig.add_trace(go.Scatter3d(
                x=[x_coords[i], x_coords[i+1]], 
                y=[y_coords[i], y_coords[i+1]], 
                z=[z_coords[i], z_coords[i+1]],
                mode='lines',
                line=dict(color=colors[i], width=18),
                showlegend=False,
                hoverinfo='skip',
                opacity=0.8
            ))
        elif ss == 'S':  # Sheets - flat ribbons
            fig.add_trace(go.Scatter3d(
                x=[x_coords[i], x_coords[i+1]], 
                y=[y_coords[i], y_coords[i+1]], 
                z=[z_coords[i], z_coords[i+1]],
                mode='lines',
                line=dict(color=colors[i], width=14),
                showlegend=False,
                hoverinfo='skip',
                opacity=0.7
            ))
    
    # 3. Side chains as smaller spheres
    side_chain_coords = []
    side_chain_colors = []
    for i, (aa, x, y, z) in enumerate(zip(sequence, x_coords, y_coords, z_coords)):
        # Offset for side chain
        offset_angle = np.random.uniform(0, 2*np.pi)
        offset_distance = 1.8
        
        sc_x = x + offset_distance * np.cos(offset_angle)
        sc_y = y + offset_distance * np.sin(offset_angle)
        sc_z = z + np.random.uniform(-0.8, 0.8)
        
        side_chain_coords.append((sc_x, sc_y, sc_z))
        side_chain_colors.append(get_aa_properties(aa)[0])
    
    sc_x, sc_y, sc_z = zip(*side_chain_coords)
    
    fig.add_trace(go.Scatter3d(
        x=sc_x, y=sc_y, z=sc_z,
        mode='markers',
        marker=dict(
            size=4,
            color=side_chain_colors,
            opacity=0.6,
            line=dict(color='darkgray', width=0.5)
        ),
        name='Side Chains',
        showlegend=False,
        hovertemplate='<b>Side Chain</b><br>%{customdata}<extra></extra>',
        customdata=[f"{aa}{i+1}" for i, aa in enumerate(sequence)]
    ))
    
    # 4. PTM sites with dramatic highlighting
    if site_positions:
        ptm_x = [x_coords[i] for i in site_positions]
        ptm_y = [y_coords[i] for i in site_positions]
        ptm_z = [z_coords[i] for i in site_positions]
        ptm_labels = [f"{sequence[i]}{i+1}" for i in site_positions]
        
        # Large red spheres for PTM sites
        fig.add_trace(go.Scatter3d(
            x=ptm_x, y=ptm_y, z=ptm_z,
            mode='markers+text',
            marker=dict(
                size=30,
                color='red',
                opacity=1.0,
                line=dict(color='darkred', width=4),
                symbol='circle'
            ),
            text=ptm_labels,
            textposition="top center",
            textfont=dict(color='red', size=18, family="Arial Black"),
            name='🔴 ADP-ribosylation Sites',
            hovertemplate='<b>🔴 ADP-ribosylation Site</b><br>' +
                         'Residue: %{text}<br>' +
                         'Critical modification site<br>' +
                         'Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>',
        ))
        
        # Glowing effect with multiple layers
        for glow_size, glow_opacity in [(45, 0.3), (60, 0.15), (75, 0.08)]:
            fig.add_trace(go.Scatter3d(
                x=ptm_x, y=ptm_y, z=ptm_z,
                mode='markers',
                marker=dict(
                    size=glow_size,
                    color='red',
                    opacity=glow_opacity,
                    line=dict(color='red', width=0)
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Professional molecular viewer styling
    fig.update_layout(
        title=dict(
            text=f'🧬 Advanced Protein Structure Visualization<br>' +
                 f'<sub>{len(sequence)} residues • {len(predicted_sites)} ADP-ribosylation sites • ' +
                 f'{len([s for s in secondary_structure if s == "H"])} helical • ' +
                 f'{len([s for s in secondary_structure if s == "S"])} sheet residues</sub>',
            x=0.5,
            font=dict(size=16, family="Arial")
        ),
        scene=dict(
            bgcolor='rgba(240,248,255,1)',
            xaxis=dict(
                title='X (Å)',
                backgroundcolor='rgba(240,248,255,0.8)',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='gray'
            ),
            yaxis=dict(
                title='Y (Å)',
                backgroundcolor='rgba(240,248,255,0.8)',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='gray'
            ),
            zaxis=dict(
                title='Z (Å)',
                backgroundcolor='rgba(240,248,255,0.8)',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='gray'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="darkblue",
            borderwidth=2,
            font=dict(size=12)
        ),
        font=dict(family="Arial", color='darkblue'),
        paper_bgcolor='white'
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
                    output_structure = gr.Plot(label="Advanced Molecular Visualization")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_analysis, output_sequence, output_structure]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 