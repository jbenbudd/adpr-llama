#!/usr/bin/env python3
"""
Simple Gradio app for testing - no model loading
"""

import gradio as gr

def predict_adpr_sites(user_sequence: str):
    """Simple test function that doesn't use any model"""
    if not user_sequence.strip():
        return "Please enter a sequence", "<div>No visualization</div>"
    
    # Just return a test response
    highlighted = f"Test sequence: {user_sequence[:50]}..."
    ngl_html = f"<div style='padding: 20px; background: #f0f0f0;'>Test visualization for: {user_sequence[:20]}...</div>"
    
    return highlighted, ngl_html

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="adpr-llama") as demo:
    gr.Markdown("# 🧬 adpr-llama – ADP-ribosylation Site Predictor")
    gr.Markdown("*Simple test version - no model loading*")
    
    with gr.Row():
        with gr.Column(scale=1):
            sequence_input = gr.Textbox(
                label="Amino Acid Sequence",
                placeholder="Enter your amino acid sequence (e.g., MASVTIGPLCYRHKNQDEFWQ)",
                lines=3
            )
            
            predict_btn = gr.Button("🧬 Predict", variant="primary")
            
            gr.Examples(
                examples=[
                    ["MASVTIGPLCYRHKNQDEFWQ"],
                    ["PDLRASGGSGAGKAKKSVDKN"],
                    ["KKKKKKKKKKKKKKKKKKKKKK"]
                ],
                inputs=sequence_input
            )
        
        with gr.Column(scale=1):
            highlighted_output = gr.HTML(label="Predicted Sites")
            ngl_view = gr.HTML(label="3D Visualization")
    
    predict_btn.click(
        fn=predict_adpr_sites,
        inputs=[sequence_input],
        outputs=[highlighted_output, ngl_view]
    )

if __name__ == "__main__":
    print("Starting simple test app...")
    demo.launch(share=False) 