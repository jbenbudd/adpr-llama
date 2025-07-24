#!/usr/bin/env python3
"""
Simple adpr-llama Gradio app for ADP-ribosylation site prediction
Uses PEFT adapter model with Zero GPU support
"""

import re
from typing import List, Tuple

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import numpy as np
import spaces

# Model configuration
MODEL_REPO = "jbenbudd/ADPrLlama"
MODEL_REVISION = "bb35aa92145ba2b6eba78542ae65e7bc7bdb06bc"  # Set to a specific commit hash like "abc123def456" if needed
CHUNK_SIZE = 21
PAD_CHAR = "-"

print(f"Loading model from {MODEL_REPO}" + (f" at revision {MODEL_REVISION}" if MODEL_REVISION else ""))

# Load the PEFT model (adapter) - this will be moved to GPU functions
model = None
tokenizer = None

def load_model():
    """Load model and tokenizer - called when needed"""
    global model, tokenizer
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
    return model, tokenizer

@spaces.GPU
def generate_prediction(prompt: str) -> str:
    """Generate prediction using the model on GPU"""
    model, tokenizer = load_model()
    
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

def generate_pdb(sequence: str, predicted_sites: List[str]) -> str:
    """Generate a simple PDB string for a helical representation of the sequence"""
    # Parse site positions for highlighting
    site_positions = set()
    for site in predicted_sites:
        match = re.match(r'[A-Z](\d+)', site)
        if match:
            site_positions.add(int(match.group(1)) - 1)  # Convert to 0-based
    
    pdb_lines = ["HEADER    PROTEIN SEQUENCE VISUALIZATION"]
    
    # Simple alpha helix coordinates (1.5Å rise per residue, 100° rotation)
    for i, aa in enumerate(sequence):
        x = 2.3 * np.cos(np.radians(i * 100))
        y = 2.3 * np.sin(np.radians(i * 100))
        z = i * 1.5
        
        # Use different chain ID for PTM sites
        chain_id = "B" if i in site_positions else "A"
        
        pdb_lines.append(
            f"ATOM  {i+1:5d}  CA  {aa:3s} {chain_id}{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
    
    pdb_lines.append("END")
    return "\n".join(pdb_lines)

def create_ngl_html(sequence: str, predicted_sites: List[str]) -> str:
    """Create HTML with embedded NGL viewer"""
    import base64
    
    # Generate PDB content
    pdb_content = generate_pdb(sequence, predicted_sites)
    pdb_b64 = base64.b64encode(pdb_content.encode()).decode()
    
    # Create unique container ID to avoid conflicts
    container_id = f"ngl-container-{abs(hash(sequence)) % 10000}"
    
    # Create NGL viewer HTML with better error handling
    html = f"""
    <div style="width: 100%; height: 450px; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa;">
        <div id="{container_id}" style="width: 100%; height: 400px; margin: 25px auto;"></div>
        <div id="ngl-status" style="text-align: center; padding: 10px; color: #666;">Loading 3D visualization...</div>
    </div>
    
    <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
    <script>
        (function() {{
            try {{
                // Wait for NGL to load
                var checkNGL = setInterval(function() {{
                    if (typeof NGL !== 'undefined') {{
                        clearInterval(checkNGL);
                        initViewer();
                    }}
                }}, 100);
                
                function initViewer() {{
                    try {{
                        var stage = new NGL.Stage("{container_id}", {{
                            backgroundColor: "white",
                            quality: "medium"
                        }});
                        
                        // Decode PDB data
                        var pdbData = atob("{pdb_b64}");
                        var blob = new Blob([pdbData], {{type: "text/plain"}});
                        
                        stage.loadFile(blob, {{ext: "pdb", name: "sequence"}}).then(function(component) {{
                            try {{
                                // Main sequence as cartoon (chain A)
                                component.addRepresentation("cartoon", {{
                                    sele: "chain A",
                                    color: "lightblue",
                                    opacity: 0.8
                                }});
                                
                                // PTM sites as spheres (chain B)
                                component.addRepresentation("spacefill", {{
                                    sele: "chain B", 
                                    color: "red",
                                    scale: 2.0
                                }});
                                
                                // Add labels for PTM sites
                                component.addRepresentation("label", {{
                                    sele: "chain B",
                                    color: "black",
                                    labelType: "residue",
                                    labelSize: 1.5
                                }});
                                
                                stage.autoView();
                                document.getElementById("ngl-status").innerHTML = "3D visualization loaded successfully! Rotate with mouse.";
                                document.getElementById("ngl-status").style.color = "#28a745";
                                
                            }} catch(e) {{
                                document.getElementById("ngl-status").innerHTML = "Error rendering structure: " + e.message;
                                document.getElementById("ngl-status").style.color = "#dc3545";
                            }}
                        }}).catch(function(error) {{
                            document.getElementById("ngl-status").innerHTML = "Error loading structure: " + error;
                            document.getElementById("ngl-status").style.color = "#dc3545";
                        }});
                        
                    }} catch(e) {{
                        document.getElementById("ngl-status").innerHTML = "Error initializing viewer: " + e.message;
                        document.getElementById("ngl-status").style.color = "#dc3545";
                    }}
                }}
                
            }} catch(e) {{
                document.getElementById("ngl-status").innerHTML = "Error loading NGL library: " + e.message;
                document.getElementById("ngl-status").style.color = "#dc3545";
            }}
        }})();
    </script>
    """
    
    return html

def predict_adpr_sites(user_sequence: str):
    """Main prediction function"""
    if not user_sequence.strip():
        return "Please enter a sequence", "<div>No sequence provided</div>", "<div>No visualization</div>"
    
    # Clean and prepare sequence
    clean_seq = clean_sequence(user_sequence)
    if not clean_seq:
        return "Invalid sequence. Please enter amino acid letters only.", "<div>Invalid sequence</div>", "<div>Invalid sequence</div>"
    
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
    
    # 3D visualization
    if len(clean_seq) <= 200:  # Only for reasonable sequence lengths
        ngl_viz = create_ngl_html(clean_seq, all_sites)
    else:
        ngl_viz = f"<div style='padding: 20px; text-align: center;'>Sequence too long for 3D visualization ({len(clean_seq)} residues). Maximum: 200 residues.</div>"
    
    return highlighted, analysis, ngl_viz

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="adpr-llama") as demo:
    gr.Markdown("# 🧬 adpr-llama – ADP-ribosylation Site Predictor")
    gr.Markdown("Enter an amino acid sequence to predict ADP-ribosylation sites. Predicted sites are shown in red in the 3D visualization.")
    
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
                    ["MASVTIGPLCYRHKNQDEFWQ"],
                    ["PDLRASGGSGAGKAKKSVDKN"],
                    ["KKKKKKKKKKKKKKKKKKKKKK"]
                ],
                inputs=sequence_input
            )
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Predicted Sites"):
                    output_sites = gr.HTML(label="Results")
                with gr.TabItem("Analysis"):
                    output_analysis = gr.HTML(label="Analysis")
                with gr.TabItem("3D Visualization"):
                    output_viz = gr.HTML(label="3D Structure")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_analysis, output_viz]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 