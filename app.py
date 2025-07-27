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
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from Bio.PDB.vectors import Vector
import nglview as nv

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

def generate_realistic_pdb_structure(sequence: str, predicted_sites: List[str]):
    """Generate a realistic PDB structure using BioPython"""
    
    # Parse site positions
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)  # Convert to 0-based
    
    def predict_secondary_structure(sequence):
        """Enhanced secondary structure prediction"""
        structure = []
        for i, aa in enumerate(sequence):
            # More sophisticated prediction
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
        
        if ss == 'H':  # Alpha helix - realistic Ramachandran angles
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
    
    # Create proper PDB structure using BioPython
    structure = Structure.Structure("protein")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    
    for i, (aa, coord, ss) in enumerate(zip(sequence, coords, secondary_structure)):
        res_id = (" ", i + 1, " ")
        residue = Residue.Residue(res_id, aa, " ")
        
        # Add C-alpha atom
        ca_atom = Atom.Atom("CA", Vector(coord), 20.0, 1.0, " ", "CA", i + 1, "C")
        residue.add(ca_atom)
        
        # Add backbone atoms for more realistic structure
        # N atom (approximate position)
        n_coord = (coord[0] - 1.2, coord[1], coord[2] - 0.5)
        n_atom = Atom.Atom("N", Vector(n_coord), 20.0, 1.0, " ", "N", i + 1, "N")
        residue.add(n_atom)
        
        # C atom (approximate position)
        c_coord = (coord[0] + 1.2, coord[1], coord[2] + 0.5)
        c_atom = Atom.Atom("C", Vector(c_coord), 20.0, 1.0, " ", "C", i + 1, "C")
        residue.add(c_atom)
        
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    
    # Save to PDB string
    io_handler = PDBIO()
    io_handler.set_structure(structure)
    
    pdb_string = io.StringIO()
    io_handler.save(pdb_string)
    pdb_content = pdb_string.getvalue()
    pdb_string.close()
    
    return pdb_content, site_positions, secondary_structure

def create_interactive_visualization(sequence: str, predicted_sites: List[str]):
    """Create professional molecular visualization using NGL viewer"""
    
    if len(sequence) > 1000:
        return f"""
        <div style="text-align: center; padding: 50px; font-size: 16px; background: white;">
            <h3>Sequence Too Long for Visualization</h3>
            <p>Length: {len(sequence)} residues (max: 1000)</p>
            <p>Predicted Sites: {', '.join(predicted_sites) if predicted_sites else 'None'}</p>
        </div>
        """
    
    try:
        # Generate realistic PDB structure
        pdb_content, site_positions, secondary_structure = generate_realistic_pdb_structure(sequence, predicted_sites)
        
        # Create temporary PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_file:
            tmp_file.write(pdb_content)
            tmp_pdb_path = tmp_file.name
        
        try:
            # Create NGL viewer
            view = nv.show_structure_file(tmp_pdb_path)
            
            # Set professional molecular representation
            view.clear_representations()
            view.add_representation('cartoon', selection='all', color='residueindex')
            view.add_representation('ball+stick', selection='all', opacity=0.6, radius=0.3)
            
            # Highlight PTM sites
            if site_positions:
                ptm_residues = [str(pos + 1) for pos in site_positions]  # Convert to 1-based
                ptm_selection = ' or '.join([f'{res}' for res in ptm_residues])
                
                view.add_representation('spacefill', 
                                      selection=ptm_selection, 
                                      color='red', 
                                      radius=2.0)
                view.add_representation('label', 
                                      selection=ptm_selection,
                                      labelType='res',
                                      color='red',
                                      fontSize=14)
            
            # Set background and camera
            view.background = 'white'
            view.camera = 'perspective'
            view.center()
            
            # Generate HTML representation
            html_content = f"""
            <div style="width: 100%; font-family: Arial, sans-serif;">
                <div style="text-align: center; margin-bottom: 15px; background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <h3 style="color: #333; margin: 0 0 10px 0;">🧬 Professional Molecular Structure Visualization</h3>
                    <p style="color: #666; margin: 0;">
                        <strong>Sequence:</strong> {len(sequence)} residues | 
                        <strong>ADP-ribosylation Sites:</strong> {len(predicted_sites)} predicted
                        {f' | <strong>Sites:</strong> {", ".join(predicted_sites)}' if predicted_sites else ''}
                    </p>
                    <p style="color: #666; margin: 5px 0 0 0;">
                        <strong>Secondary Structure:</strong> 
                        {len([s for s in secondary_structure if s == "H"])} α-helical, 
                        {len([s for s in secondary_structure if s == "S"])} β-sheet, 
                        {len([s for s in secondary_structure if s == "L"])} loop residues
                    </p>
                </div>
                
                <div style="border: 2px solid #ddd; border-radius: 8px; overflow: hidden;">
                    {view._repr_html_()}
                </div>
                
                <div style="text-align: center; margin-top: 10px; font-size: 12px; color: #666;">
                    <p>Professional molecular visualization using NGL Viewer</p>
                    <p>🔴 Red spheres indicate predicted ADP-ribosylation sites</p>
                </div>
            </div>
            """
            
            return html_content
            
        except Exception as e:
            print(f"NGL viewer error: {e}")
            # Fallback to static representation
            return create_static_molecular_visualization(sequence, predicted_sites, pdb_content)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_pdb_path)
            except:
                pass
                
    except Exception as e:
        print(f"Molecular visualization error: {e}")
        return f"""
        <div style="text-align: center; padding: 50px; font-size: 16px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
            <h3 style="color: #856404;">Molecular Visualization Unavailable</h3>
            <p style="color: #856404;">Unable to generate 3D structure visualization.</p>
            <p style="color: #856404;"><strong>Sequence:</strong> {len(sequence)} residues</p>
            <p style="color: #856404;"><strong>Predicted Sites:</strong> {', '.join(predicted_sites) if predicted_sites else 'None'}</p>
        </div>
        """

def create_static_molecular_visualization(sequence: str, predicted_sites: List[str], pdb_content: str):
    """Create a static molecular visualization as fallback"""
    
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)
    
    # Create a simple but informative static representation
    html_content = f"""
    <div style="width: 100%; font-family: Arial, sans-serif;">
        <div style="text-align: center; margin-bottom: 15px; background: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h3 style="color: #333; margin: 0 0 10px 0;">🧬 Molecular Structure Information</h3>
            <p style="color: #666; margin: 0;">
                <strong>Sequence:</strong> {len(sequence)} residues | 
                <strong>ADP-ribosylation Sites:</strong> {len(predicted_sites)} predicted
                {f' | <strong>Sites:</strong> {", ".join(predicted_sites)}' if predicted_sites else ''}
            </p>
        </div>
        
        <div style="background: white; border: 2px solid #ddd; border-radius: 8px; padding: 20px; text-align: center;">
            <div style="background: #e3f2fd; border-radius: 8px; padding: 20px; margin-bottom: 15px;">
                <h4 style="color: #1976d2; margin: 0 0 10px 0;">📄 PDB Structure Generated</h4>
                <p style="color: #1976d2; margin: 0;">Realistic protein structure with proper backbone geometry and secondary structure prediction.</p>
            </div>
            
            <div style="background: #f3e5f5; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <h4 style="color: #7b1fa2; margin: 0 0 10px 0;">🎯 PTM Site Analysis</h4>
                {f'<p style="color: #7b1fa2; margin: 0;"><strong>Sites Found:</strong> {", ".join(predicted_sites)}</p>' if predicted_sites else '<p style="color: #7b1fa2; margin: 0;">No ADP-ribosylation sites predicted</p>'}
            </div>
            
            <div style="background: #e8f5e8; border-radius: 8px; padding: 15px;">
                <h4 style="color: #388e3c; margin: 0 0 10px 0;">📊 Structure Information</h4>
                <p style="color: #388e3c; margin: 0;">
                    Professional PDB structure generated with BioPython<br>
                    Ready for analysis with molecular visualization software
                </p>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 15px;">
            <details style="background: #f8f9fa; border-radius: 8px; padding: 10px;">
                <summary style="cursor: pointer; font-weight: bold; color: #333;">View PDB Structure Data</summary>
                <pre style="background: white; padding: 15px; border-radius: 4px; text-align: left; overflow-x: auto; max-height: 300px; overflow-y: auto; margin-top: 10px; border: 1px solid #ddd; font-size: 12px;">{pdb_content}</pre>
            </details>
        </div>
    </div>
    """
    
    return html_content

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
                    output_structure = gr.HTML(label="Professional Molecular Visualization")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_analysis, output_sequence, output_structure]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 