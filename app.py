#!/usr/bin/env python3
"""
Simple adpr-llama Gradio app for ADP-ribosylation site prediction
Uses PEFT adapter model with Zero GPU support
"""

import re
from typing import List, Tuple
import io
import base64
import tempfile
import os

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import numpy as np
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
    """Create simple but effective 3D protein visualization"""
    
    if len(sequence) > 1000:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Sequence too long<br>Length: {len(sequence)} residues<br>Sites: {', '.join(predicted_sites) if predicted_sites else 'None'}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16), align="center"
        )
        fig.update_layout(title="Sequence Too Long", width=800, height=400)
        return fig
    
    # Parse PTM sites
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)
    
    # Generate simple 3D coordinates
    n = len(sequence)
    coords = []
    
    # Create a simple helix-like structure
    for i in range(n):
        angle = i * 2 * np.pi / 3.6  # ~3.6 residues per turn
        radius = 2.0
        x = radius * np.cos(angle) + np.random.normal(0, 0.3)
        y = radius * np.sin(angle) + np.random.normal(0, 0.3)
        z = i * 1.5 + np.random.normal(0, 0.2)
        coords.append((x, y, z))
    
    x_coords, y_coords, z_coords = zip(*coords)
    
    # Create the plot
    fig = go.Figure()
    
    # Protein backbone
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines+markers',
        line=dict(color='blue', width=8),
        marker=dict(size=5, color='lightblue'),
        name='Protein Backbone',
        hovertemplate='<b>%{text}</b><br>Position: %{customdata}<extra></extra>',
        text=[f"{aa}{i+1}" for i, aa in enumerate(sequence)],
        customdata=[i+1 for i in range(len(sequence))]
    ))
    
    # Highlight PTM sites
    if site_positions:
        ptm_x = [x_coords[i] for i in site_positions]
        ptm_y = [y_coords[i] for i in site_positions]
        ptm_z = [z_coords[i] for i in site_positions]
        ptm_labels = [f"{sequence[i]}{i+1}" for i in site_positions]
        
        fig.add_trace(go.Scatter3d(
            x=ptm_x, y=ptm_y, z=ptm_z,
            mode='markers+text',
            marker=dict(size=20, color='red'),
            text=ptm_labels,
            textposition="top center",
            name='ADP-ribosylation Sites',
            hovertemplate='<b>🔴 PTM Site</b><br>%{text}<extra></extra>'
        ))
    
    # Simple layout
    fig.update_layout(
        title=f'3D Protein Structure - {len(sequence)} residues, {len(predicted_sites)} PTM sites',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600
    )
    
    return fig

def create_sequence_plot(sequence: str, predicted_sites: List[str]):
    """Create a 2D sequence visualization using HTML/CSS"""
    
    # Parse site positions
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)  # Convert to 0-based
    
    # Create sequence grid
    residues_per_row = 50
    rows_needed = (len(sequence) + residues_per_row - 1) // residues_per_row
    
    # Build HTML grid
    html_rows = []
    for row in range(rows_needed):
        html_cols = []
        for col in range(residues_per_row):
            seq_idx = row * residues_per_row + col
            if seq_idx < len(sequence):
                residue = sequence[seq_idx]
                is_ptm = seq_idx in site_positions
                
                style = f"""
                    display: inline-block; 
                    width: 25px; 
                    height: 25px; 
                    margin: 1px; 
                    text-align: center; 
                    line-height: 25px; 
                    font-family: monospace; 
                    font-size: 12px; 
                    font-weight: bold;
                    border-radius: 3px;
                    border: 1px solid #ddd;
                    background-color: {'#ff4444' if is_ptm else '#e8f4fd'};
                    color: {'white' if is_ptm else '#333'};
                """
                
                tooltip = f"{residue}{seq_idx + 1}" + (" - ADP-ribosylation Site" if is_ptm else "")
                html_cols.append(f'<div style="{style}" title="{tooltip}">{residue}</div>')
            else:
                html_cols.append('<div style="display: inline-block; width: 25px; height: 25px; margin: 1px;"></div>')
        
        html_rows.append(f'<div style="text-align: center; margin: 2px 0;">{"".join(html_cols)}</div>')
    
    # Create complete HTML
    html_content = f"""
    <div style="width: 100%; font-family: Arial, sans-serif;">
        <div style="text-align: center; margin-bottom: 15px; background: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h3 style="color: #333; margin: 0 0 10px 0;">📊 Sequence Layout Visualization</h3>
            <p style="color: #666; margin: 0;">
                <strong>Length:</strong> {len(sequence)} residues | 
                <strong>PTM Sites:</strong> {len(predicted_sites)} predicted
            </p>
        </div>
        
        <div style="background: white; border: 2px solid #ddd; border-radius: 8px; padding: 20px; overflow-x: auto;">
            <div style="text-align: center; margin-bottom: 15px;">
                <span style="display: inline-block; width: 20px; height: 20px; background: #e8f4fd; border: 1px solid #ddd; border-radius: 3px; margin-right: 5px; vertical-align: middle;"></span>
                <span style="color: #666; margin-right: 20px;">Normal Residue</span>
                <span style="display: inline-block; width: 20px; height: 20px; background: #ff4444; border: 1px solid #ddd; border-radius: 3px; margin-right: 5px; vertical-align: middle;"></span>
                <span style="color: #666;">ADP-ribosylation Site</span>
            </div>
            
            {"".join(html_rows)}
            
            {f'<div style="margin-top: 15px; text-align: center; font-size: 14px; color: #666;"><strong>Predicted Sites:</strong> {", ".join(predicted_sites)}</div>' if predicted_sites else '<div style="margin-top: 15px; text-align: center; font-size: 14px; color: #666;">No ADP-ribosylation sites predicted</div>'}
        </div>
    </div>
    """
    
    return html_content

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
                    output_sequence = gr.HTML(label="2D Sequence Visualization")
                with gr.TabItem("3D Structure"):
                    output_structure = gr.Plot(label="3D Structure")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_analysis, output_sequence, output_structure]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 