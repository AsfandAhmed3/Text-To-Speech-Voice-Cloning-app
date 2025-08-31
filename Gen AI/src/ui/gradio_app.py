"""
Gradio Web Interface for PDF to Speech with Voice Cloning

This module provides a user-friendly web interface using Gradio
for the PDF to speech conversion and voice cloning functionality.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Please run: pip install gradio")

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import PDFToSpeechApp

logger = logging.getLogger(__name__)


class GradioInterface:
    """Gradio web interface for PDF to Speech application."""
    
    def __init__(self):
        """Initialize the Gradio interface."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is not available")
        
        self.app = PDFToSpeechApp()
        self.temp_dir = Path(tempfile.gettempdir()) / "pdf_to_speech"
        self.temp_dir.mkdir(exist_ok=True)
        
    def upload_pdf(self, pdf_file) -> str:
        """Handle PDF file upload."""
        if pdf_file is None:
            return "No PDF file uploaded"
        
        try:
            # Save uploaded file to temp directory
            pdf_path = self.temp_dir / f"uploaded_{Path(pdf_file.name).name}"
            
            # Copy the uploaded file
            import shutil
            shutil.copy2(pdf_file.name, pdf_path)
            
            # Get PDF info
            pdf_info = self.app.pdf_extractor.get_pdf_info(pdf_path)
            
            info_text = f"""
**PDF Information:**
- Filename: {pdf_info['file_name']}
- File size: {pdf_info['file_size_mb']} MB
- Total pages: {pdf_info['total_pages']}
- Estimated words: {pdf_info.get('estimated_words', 'Unknown')}
- Estimated reading time: {pdf_info.get('estimated_words', 0) / 150:.1f} minutes
            """
            
            return info_text
            
        except Exception as e:
            logger.error(f"Error processing PDF upload: {e}")
            return f"Error processing PDF: {e}"
    
    def upload_voice_sample(self, voice_files) -> str:
        """Handle voice sample file upload."""
        if not voice_files:
            return "No voice samples uploaded"
        
        try:
            uploaded_samples = []
            
            for i, voice_file in enumerate(voice_files):
                if voice_file is None:
                    continue
                
                # Save to temp directory
                sample_path = self.temp_dir / f"voice_sample_{i}_{Path(voice_file.name).name}"
                
                import shutil
                shutil.copy2(voice_file.name, sample_path)
                uploaded_samples.append(sample_path)
                
                # Validate the sample
                validation = self.app.voice_cloner.validate_voice_sample(sample_path)
                
                info = f"""
**Voice Sample {i+1}:**
- Filename: {Path(voice_file.name).name}
- Valid: {'✓' if validation['valid'] else '✗'}
- Duration: {validation.get('duration', 'Unknown'):.1f} seconds
- Sample rate: {validation.get('sample_rate', 'Unknown')} Hz
- File size: {validation.get('file_size', 0) / (1024*1024):.1f} MB
                """
                
                if validation['issues']:
                    info += f"\\n- Issues: {', '.join(validation['issues'])}"
                
                return info
            
            return f"Uploaded {len(uploaded_samples)} voice samples successfully"
            
        except Exception as e:
            logger.error(f"Error processing voice samples: {e}")
            return f"Error processing voice samples: {e}"
    
    def process_pdf_to_speech(self,
                             pdf_file,
                             voice_files,
                             voice_name: str,
                             extract_chapters: bool,
                             tts_engine: str,
                             chunk_size: int,
                             progress=gr.Progress()) -> Tuple[str, str]:
        """Process the full PDF to speech workflow."""
        if pdf_file is None:
            return "Error: No PDF file provided", ""
        
        if not voice_files:
            return "Error: No voice samples provided", ""
        
        if not voice_name.strip():
            voice_name = "ClonedVoice"
        
        try:
            progress(0.1, desc="Preparing files...")
            
            # Prepare file paths
            pdf_path = self.temp_dir / f"uploaded_{Path(pdf_file.name).name}"
            voice_sample_paths = []
            
            for i, voice_file in enumerate(voice_files):
                if voice_file is not None:
                    sample_path = self.temp_dir / f"voice_sample_{i}_{Path(voice_file.name).name}"
                    voice_sample_paths.append(str(sample_path))
            
            # Update app configuration
            self.app.config.update({
                'tts_engine': tts_engine.lower(),
                'voice_cloning_service': tts_engine.lower(),
                'chunk_size': chunk_size
            })
            
            progress(0.2, desc="Extracting text from PDF...")
            
            # Run the complete workflow
            results = self.app.process_full_workflow(
                pdf_path=str(pdf_path),
                voice_sample_paths=voice_sample_paths,
                voice_name=voice_name,
                extract_by_chapters=extract_chapters
            )
            
            progress(1.0, desc="Processing complete!")
            
            # Generate summary report
            report = self.app.get_summary_report(results)
            
            # Generate download links
            download_info = self._generate_download_info(results)
            
            return report, download_info
            
        except Exception as e:
            logger.error(f"Error in PDF to speech processing: {e}")
            return f"Error processing: {e}", ""
    
    def _generate_download_info(self, results: dict) -> str:
        """Generate download information for the results."""
        download_info = ["\\n### Generated Files:\\n"]
        
        if 'default_voice_files' in results:
            download_info.append("**Default Voice Audio Files:**")
            for chapter, files in results['default_voice_files'].items():
                download_info.append(f"- {chapter}: {len(files)} files")
        
        if 'cloned_voice_files' in results:
            download_info.append("\\n**Cloned Voice Audio Files:**")
            for chapter, files in results['cloned_voice_files'].items():
                download_info.append(f"- {chapter}: {len(files)} files")
        
        download_info.append(f"\\n📁 All files saved to: `{self.app.config['output_dir']}`")
        download_info.append("\\n💡 **Tip:** Check the output directory for all generated audio files!")
        
        return "\\n".join(download_info)
    
    def create_interface(self) -> gr.Interface:
        """Create and return the Gradio interface."""
        
        with gr.Blocks(title="PDF to Speech with Voice Cloning", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 📚🎵 PDF to Speech with Voice Cloning
            
            Convert your PDF books into audiobooks with your own cloned voice!
            
            ## How to use:
            1. **Upload a PDF** - Select the book you want to convert
            2. **Upload voice samples** - Record 1-2 minutes of clear speech
            3. **Configure settings** - Choose your preferences
            4. **Process** - Wait for the magic to happen!
            
            ---
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # PDF Upload Section
                    gr.Markdown("### 📄 PDF Upload")
                    pdf_input = gr.File(
                        label="Upload PDF Book",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    pdf_info = gr.Textbox(
                        label="PDF Information",
                        interactive=False,
                        max_lines=10
                    )
                    
                    # Voice Sample Upload Section
                    gr.Markdown("### 🎤 Voice Samples")
                    voice_input = gr.File(
                        label="Upload Voice Samples (1-3 files, 30s-5min each)",
                        file_count="multiple",
                        file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                        type="filepath"
                    )
                    voice_info = gr.Textbox(
                        label="Voice Sample Information",
                        interactive=False,
                        max_lines=8
                    )
                
                with gr.Column(scale=1):
                    # Configuration Section
                    gr.Markdown("### ⚙️ Configuration")
                    
                    voice_name_input = gr.Textbox(
                        label="Voice Name",
                        value="MyClonedVoice",
                        placeholder="Enter a name for your cloned voice"
                    )
                    
                    extract_chapters = gr.Checkbox(
                        label="Extract by Chapters",
                        value=True,
                        info="Split the book into chapters for easier processing"
                    )
                    
                    tts_engine = gr.Dropdown(
                        label="TTS Engine",
                        choices=["Coqui", "ElevenLabs", "PyTTSX3"],
                        value="Coqui",
                        info="Choose the text-to-speech engine (Coqui recommended)"
                    )
                    
                    chunk_size = gr.Slider(
                        label="Chunk Size (characters)",
                        minimum=500,
                        maximum=2000,
                        value=1000,
                        step=100,
                        info="Size of text chunks for processing"
                    )
                    
                    # Process Button
                    process_btn = gr.Button(
                        "🚀 Start Processing",
                        variant="primary",
                        size="lg"
                    )
            
            # Results Section
            gr.Markdown("### 📊 Processing Results")
            
            with gr.Row():
                results_output = gr.Textbox(
                    label="Processing Summary",
                    interactive=False,
                    max_lines=20
                )
                
                download_output = gr.Textbox(
                    label="Download Information",
                    interactive=False,
                    max_lines=15
                )
            
            # Event handlers
            pdf_input.change(
                fn=self.upload_pdf,
                inputs=[pdf_input],
                outputs=[pdf_info]
            )
            
            voice_input.change(
                fn=self.upload_voice_sample,
                inputs=[voice_input],
                outputs=[voice_info]
            )
            
            process_btn.click(
                fn=self.process_pdf_to_speech,
                inputs=[
                    pdf_input,
                    voice_input,
                    voice_name_input,
                    extract_chapters,
                    tts_engine,
                    chunk_size
                ],
                outputs=[results_output, download_output]
            )
            
            # Footer
            gr.Markdown("""
            ---
            
            ### 📝 Notes:
            - **Voice Samples**: For best results, use clear, noise-free recordings
            - **Processing Time**: Depends on PDF length and TTS engine
            - **API Keys**: Set up your ElevenLabs API key in environment variables
            - **Output**: Files are saved in the `output/` directory
            
            ### 🔧 Troubleshooting:
            - PDF not working? Try a text-based PDF (not scanned images)
            - Voice cloning issues? Ensure samples are high quality and 30s+ long
            - Processing slow? Try smaller chunk sizes or fewer chapters
            
            **Made with ❤️ for Generative AI learning**
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is not available")
        
        interface = self.create_interface()
        
        # Default launch parameters
        launch_kwargs = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'debug': False
        }
        launch_kwargs.update(kwargs)
        
        logger.info(f"Launching Gradio interface on http://localhost:{launch_kwargs['server_port']}")
        interface.launch(**launch_kwargs)


def main():
    """Main entry point for the Gradio interface."""
    if not GRADIO_AVAILABLE:
        print("Error: Gradio is not installed.")
        print("Please install it with: pip install gradio")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create and launch interface
        gradio_app = GradioInterface()
        gradio_app.launch(
            share=False,  # Set to True to create a public link
            debug=True if os.getenv('DEBUG') else False
        )
        
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
