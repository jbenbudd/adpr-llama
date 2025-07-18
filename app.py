#!/usr/bin/env python3
"""
Simple adpr-llama Gradio app for ADP-ribosylation site prediction
Uses PEFT adapter model on CPU with 32GB RAM
"""

import re
from typing import List, Tuple

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

# Model configuration
MODEL_REPO = "jbenbudd/ADPrLlama"
MODEL_REVISION = "bb35aa92145ba2b6eba78542ae65e7bc7bdb06bc"  # Set to a specific commit hash like "abc123def456" if needed
CHUNK_SIZE = 21
PAD_CHAR = "-"

print(f"Loading model from {MODEL_REPO}" + (f" at revision {MODEL_REVISION}" if MODEL_REVISION else ""))

# Load the PEFT model (adapter)
model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_REPO,
    revision=MODEL_REVISION,  # This allows specifying a commit hash
    device_map="cpu",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO, 
    revision=MODEL_REVISION,  # Also use same revision for tokenizer
    use_fast=True
)

print("Model loaded successfully!")

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

def predict_adpr_sites(user_sequence: str):
    """Main prediction function"""
    if not user_sequence.strip():
        return "Please enter a sequence", "<div>No sequence provided</div>"
    
    # Clean and prepare sequence
    clean_seq = clean_sequence(user_sequence)
    if not clean_seq:
        return "Invalid sequence. Please enter amino acid letters only.", "<div>Invalid sequence</div>"
    
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
        inputs = tokenizer(prompt, return_tensors="pt")
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
        
        print(f"Raw response: {response}")
        
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
    
    # Simple visualization
    visualization = f"""
    <div style="padding: 20px; background: #f5f5f5; border-radius: 8px;">
        <h4>Sequence Analysis</h4>
        <p><strong>Original length:</strong> {original_length} residues</p>
        <p><strong>Chunks processed:</strong> {len(chunks)}</p>
        <p><strong>Sites found:</strong> {len(all_sites)}</p>
    </div>
    """
    
    return highlighted, visualization

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="adpr-llama") as demo:
    gr.Markdown("# 🧬 adpr-llama – ADP-ribosylation Site Predictor")
    gr.Markdown("Enter an amino acid sequence to predict ADP-ribosylation sites.")
    
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
            output_sites = gr.HTML(label="Predicted Sites")
            output_viz = gr.HTML(label="Analysis")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[output_sites, output_viz]
    )

if __name__ == "__main__":
    print("Starting adpr-llama app...")
    demo.launch(share=False) 