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
def generate_prediction(prompt: str, request: gr.Request = None) -> str:
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
        if "quota" in str(e).lower() or "authentication" in str(e).lower():
            raise gr.Error("Authentication required. Please log in to access GPU resources.")
        else:
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
    """Create interactive 3D visualization using Plotly"""
    
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
    
    # Generate 3D coordinates for protein backbone
    x_coords = []
    y_coords = []
    z_coords = []
    
    for i in range(len(sequence)):
        # Create a more interesting 3D structure - alpha helix-like
        angle = i * 100  # degrees per residue
        radius = 2.3  # helix radius
        rise = 1.5   # rise per residue
        
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        z = i * rise
        
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
    
    # Create the plot
    fig = go.Figure()
    
    # Add protein backbone as a line
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines+markers',
        line=dict(color='lightblue', width=8),
        marker=dict(size=4, color='lightblue', opacity=0.8),
        name='Protein Backbone',
        hovertemplate='<b>Position %{text}</b><br>Residue: %{customdata}<extra></extra>',
        text=[i+1 for i in range(len(sequence))],
        customdata=[aa for aa in sequence]
    ))
    
    # Add PTM sites as large red spheres
    if site_positions:
        ptm_x = [x_coords[i] for i in site_positions]
        ptm_y = [y_coords[i] for i in site_positions]
        ptm_z = [z_coords[i] for i in site_positions]
        ptm_labels = [f"{sequence[i]}{i+1}" for i in site_positions]
        ptm_residues = [sequence[i] for i in site_positions]
        
        fig.add_trace(go.Scatter3d(
            x=ptm_x, y=ptm_y, z=ptm_z,
            mode='markers+text',
            marker=dict(
                size=15,
                color='red',
                opacity=1.0,
                line=dict(color='darkred', width=2)
            ),
            text=ptm_labels,
            textposition="top center",
            textfont=dict(color='red', size=12),
            name='ADP-ribosylation Sites',
            hovertemplate='<b>ADP-ribosylation Site</b><br>Position: %{text}<br>Residue: %{customdata}<extra></extra>',
            customdata=ptm_residues
        ))
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text=f'Interactive 3D Protein Structure<br><sub>{len(sequence)} residues, {len(predicted_sites)} ADP-ribosylation sites</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
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

def predict_adpr_sites(user_sequence: str, request: gr.Request = None):
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
        
        # Generate prediction (pass request for proper authentication)
        response = generate_prediction(prompt, request)
        
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
with gr.Blocks(theme=gr.themes.Soft(), title="adpr-llama") as demo:
    gr.Markdown("# 🧬 adpr-llama – ADP-ribosylation Site Predictor")
    gr.Markdown("Enter an amino acid sequence to predict ADP-ribosylation sites. Predicted sites are highlighted in red in both sequence and 3D visualizations.")
    
    # Add login button for proper authentication (required for ZeroGPU)
    gr.Markdown("⚠️ **Please log in to use this space.** This app requires GPU resources and authentication for quota management.")
    gr.LoginButton()
    
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
    demo.launch(
        share=False,
        show_error=True,
        auth=None,  # Use HF authentication
    ) 