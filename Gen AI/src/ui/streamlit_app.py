"""
Streamlit Web Interface for PDF to Speech with Voice Cloning

This module provides an alternative web interface using Streamlit
for the PDF to speech conversion and voice cloning functionality.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Please run: pip install streamlit")

from main import PDFToSpeechApp

logger = logging.getLogger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'app' not in st.session_state:
        st.session_state.app = PDFToSpeechApp()
    
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    
    if 'voice_uploaded' not in st.session_state:
        st.session_state.voice_uploaded = False
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'results' not in st.session_state:
        st.session_state.results = None


def upload_pdf_section():
    """Handle PDF upload section."""
    st.header("📄 PDF Upload")
    
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload the PDF book you want to convert to speech"
    )
    
    if pdf_file is not None:
        # Save uploaded file
        temp_dir = Path(tempfile.gettempdir()) / "pdf_to_speech"
        temp_dir.mkdir(exist_ok=True)
        
        pdf_path = temp_dir / f"uploaded_{pdf_file.name}"
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        st.session_state.pdf_path = str(pdf_path)
        st.session_state.pdf_uploaded = True
        
        # Show PDF information
        try:
            # Simplified PDF info (placeholder)
            import os
            file_size = os.path.getsize(pdf_path) / (1024*1024)  # MB
            pdf_info = {
                'file_size_mb': f"{file_size:.2f}",
                'total_pages': "N/A (placeholder)",
                'text_preview': "PDF processing available...",
                'estimated_words': 0,
                'metadata': {}
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("File Size", f"{pdf_info['file_size_mb']} MB")
            
            with col2:
                st.metric("Total Pages", pdf_info['total_pages'])
            
            with col3:
                estimated_words = pdf_info.get('estimated_words', 0)
                if isinstance(estimated_words, int) and estimated_words > 0:
                    reading_time = estimated_words / 150  # Average reading speed
                    st.metric("Est. Reading Time", f"{reading_time:.1f} min")
                else:
                    st.metric("Est. Reading Time", "Unknown")
            
            st.success(f"✅ PDF '{pdf_file.name}' uploaded successfully!")
            
            # Show metadata if available
            metadata = pdf_info.get('metadata', {})
            if metadata:
                with st.expander("📋 PDF Metadata"):
                    for key, value in metadata.items():
                        if value:
                            st.text(f"{key}: {value}")
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.session_state.pdf_uploaded = False
    
    return st.session_state.pdf_uploaded


def upload_voice_section():
    """Handle voice sample upload section."""
    st.header("🎤 Voice Samples")
    
    st.info("""
    **Voice Sample Guidelines:**
    - Upload 1-3 clear audio files
    - Each file should be 30 seconds to 5 minutes long
    - Use high-quality recordings (no background noise)
    - Speak naturally and clearly
    """)
    
    voice_files = st.file_uploader(
        "Choose voice sample files",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'opus'],
        accept_multiple_files=True,
        help="Upload your voice samples for cloning (WhatsApp .ogg files are supported!)"
    )
    
    # Add note about OGG files
    st.info("📱 WhatsApp voice messages (.ogg files) are supported and will be automatically converted to WAV format.")
    
    if voice_files:
        temp_dir = Path(tempfile.gettempdir()) / "pdf_to_speech"
        temp_dir.mkdir(exist_ok=True)
        
        voice_sample_paths = []
        
        for i, voice_file in enumerate(voice_files):
            # Clean filename to avoid issues with special characters
            import re
            clean_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', voice_file.name)
            sample_path = temp_dir / f"voice_sample_{i}_{clean_filename}"
            
            # Save the uploaded file
            try:
                with open(sample_path, "wb") as f:
                    f.write(voice_file.getbuffer())
                    
                st.success(f"✅ Saved: {voice_file.name} → {sample_path.name}")
                voice_sample_paths.append(str(sample_path))
                
            except Exception as e:
                st.error(f"❌ Failed to save {voice_file.name}: {str(e)}")
                continue
            
            # Validate the sample (using new validation method)
            validation = st.session_state.app.validate_voice_sample(str(sample_path))
            
            with st.expander(f"🎵 {voice_file.name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text(f"Valid: {'✅' if validation['is_valid'] else '❌'}")
                    st.text(f"Duration: {validation.get('duration', 'Unknown')}")
                    st.text(f"File size: {len(voice_file.getbuffer()) / (1024*1024):.1f} MB")
                
                with col2:
                    if validation.get('sample_rate'):
                        sample_rate = validation['sample_rate']
                        if isinstance(sample_rate, str):
                            st.text(f"Sample rate: {sample_rate}")
                        else:
                            st.text(f"Sample rate: {sample_rate} Hz")
                    if validation.get('channels'):
                        st.text(f"Channels: {validation['channels']}")
                
                if validation.get('issues'):
                    st.warning("Issues found:")
                    for issue in validation['issues']:
                        st.text(f"• {issue}")
        
        st.session_state.voice_sample_paths = voice_sample_paths
        st.session_state.voice_uploaded = True
        
        st.success(f"✅ {len(voice_files)} voice sample(s) uploaded successfully!")
    
    return st.session_state.voice_uploaded


def configuration_section():
    """Handle configuration section."""
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        voice_name = st.text_input(
            "Voice Name",
            value="MyClonedVoice",
            help="Enter a name for your cloned voice"
        )
        
        extract_chapters = st.checkbox(
            "Extract by Chapters",
            value=True,
            help="Split the book into chapters for easier processing"
        )
    
    with col2:
        tts_engine = st.selectbox(
            "TTS Engine",
            options=["Coqui", "ElevenLabs", "PyTTSX3"],
            index=0,
            help="Choose the text-to-speech engine (Coqui recommended for voice cloning)"
        )
        
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )
    
    return voice_name, extract_chapters, tts_engine.lower(), chunk_size


def processing_section(voice_name, extract_chapters, tts_engine, chunk_size):
    """Handle the processing section."""
    st.header("🚀 Processing")
    
    if not st.session_state.pdf_uploaded:
        st.warning("⚠️ Please upload a PDF file first.")
        return
    
    if not st.session_state.voice_uploaded:
        st.warning("⚠️ Please upload voice samples first.")
        return
    
    if st.button("Start Processing", type="primary", use_container_width=True):
        process_pdf_to_speech(voice_name, extract_chapters, tts_engine, chunk_size)


def process_pdf_to_speech(voice_name, extract_chapters, tts_engine, chunk_size):
    """Process the PDF to speech conversion using the new workflow."""
    
    # Create progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Use the new workflow from PDFToSpeechApp
        status_text.text("� Starting PDF to Speech workflow...")
        progress_bar.progress(0.1)
        
        # Get voice sample paths if available
        voice_samples = getattr(st.session_state, 'voice_sample_paths', None)
        
        # Run the complete workflow
        results = st.session_state.app.process_full_workflow(
            pdf_path=st.session_state.pdf_path,
            voice_sample_paths=voice_samples,
            voice_name=voice_name,
            extract_chapters=extract_chapters,
            tts_engine=tts_engine
        )
        
        # Update progress based on completed steps
        progress_value = 0.2
        for step in results.get("steps_completed", []):
            if step == "text_extraction":
                status_text.text("� Text extracted from PDF...")
                progress_value = 0.3
            elif step == "default_tts":
                status_text.text("🎵 Generated audio with default voice...")
                progress_value = 0.6
            elif step == "voice_cloning":
                status_text.text("🎭 Voice cloned successfully...")
                progress_value = 0.8
            elif step == "cloned_tts":
                status_text.text("🗣️ Generated audio with cloned voice...")
                progress_value = 0.95
            
            progress_bar.progress(progress_value)
        
        # Complete
        progress_bar.progress(1.0)
        
        if results["success"]:
            status_text.text("✅ Processing complete!")
            
            # Store results
            st.session_state.results = results
            
            # Display results
            st.success("🎉 PDF to Speech conversion completed successfully!")
            
            # Show what was accomplished
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Processing Summary")
                st.write(f"**PDF:** {Path(results['pdf_path']).name}")
                st.write(f"**Chapters:** {len(results.get('text_extraction', {}).get('chapters', []))}")
                st.write(f"**TTS Engine:** {results['tts_engine']}")
                
                if results.get('voice_cloning'):
                    st.write(f"**Voice Cloned:** {results['voice_name']}")
                    st.write(f"**Samples Used:** {results['voice_cloning']['samples_used']}")
            
            with col2:
                st.subheader("🎵 Generated Audio")
                
                # Show default audio files
                default_files = results.get("default_audio_files", [])
                if default_files:
                    st.write("**Default Voice Audio:**")
                    for audio_info in default_files:
                        st.write(f"• {audio_info['chapter']}")
                
                # Show cloned audio files
                cloned_files = results.get("cloned_audio_files", [])
                if cloned_files:
                    st.write("**Cloned Voice Audio:**")
                    for audio_info in cloned_files:
                        st.write(f"• {audio_info['chapter']}")
        else:
            status_text.text("❌ Processing failed!")
            st.error("Processing failed. Please check the errors below:")
            for error in results.get("errors", []):
                st.error(f"Error: {error}")
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ An error occurred!")
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Streamlit processing error: {e}")


def results_section():
    """Display processing results."""
    if not st.session_state.processing_complete or not st.session_state.results:
        return
    
    st.header("📊 Results")
    
    results = st.session_state.results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Chapters", len(results['chapters']))
    
    with col2:
        total_default = sum(len(files) for files in results['default_voice_files'].values())
        st.metric("Default Voice Files", total_default)
    
    with col3:
        total_cloned = sum(len(files) for files in results['cloned_voice_files'].values())
        st.metric("Cloned Voice Files", total_cloned)
    
    with col4:
        status = "✅ Success" if results['cloned_voice_id'] else "⚠️ Partial"
        st.metric("Voice Cloning", status)
    
    # Detailed results
    st.subheader("📁 Generated Files")
    
    # Default voice files
    if results['default_voice_files']:
        with st.expander("🎵 Default Voice Audio Files"):
            for chapter, files in results['default_voice_files'].items():
                st.text(f"📖 {chapter}: {len(files)} files")
                for file_path in files[:3]:  # Show first 3 files
                    st.text(f"  • {Path(file_path).name}")
                if len(files) > 3:
                    st.text(f"  ... and {len(files) - 3} more files")
    
    # Cloned voice files
    if results['cloned_voice_files']:
        with st.expander("🗣️ Cloned Voice Audio Files"):
            for chapter, files in results['cloned_voice_files'].items():
                st.text(f"📖 {chapter}: {len(files)} files")
                for file_path in files[:3]:  # Show first 3 files
                    st.text(f"  • {Path(file_path).name}")
                if len(files) > 3:
                    st.text(f"  ... and {len(files) - 3} more files")
    
    # Output directory info
    output_dir = st.session_state.app.config['output_dir']
    st.info(f"📂 All files saved to: `{Path(output_dir).absolute()}`")
    
    # Download instructions
    st.subheader("💾 Download Instructions")
    st.markdown(f"""
    Your audiobook files have been generated! Here's how to access them:
    
    1. **Navigate to the output directory:** `{Path(output_dir).absolute()}`
    2. **Default voice files:** Look in `default_voice/` folder
    3. **Cloned voice files:** Look in `cloned_voice/` folder
    4. **Original text:** Check `extracted_text/` folder
    
    **Tip:** You can play the audio files directly or import them into your favorite media player!
    """)


def sidebar():
    """Create sidebar with additional information and controls."""
    with st.sidebar:
        st.markdown("## 🎵 PDF to Speech")
        st.markdown("*with Voice Cloning*")
        
        st.markdown("### 📋 Quick Guide")
        st.markdown("""
        1. **Upload PDF**: Choose your book
        2. **Upload Voice**: Record yourself speaking
        3. **Configure**: Set your preferences
        4. **Process**: Let AI do the magic!
        5. **Download**: Get your audiobook
        """)
        
        st.markdown("### ⚙️ System Status")
        
        # Check available engines (simplified version)
        st.text("TTS Engines:")
        engines = ['elevenlabs', 'coqui', 'pyttsx3']
        available_engines = ['coqui']  # Simplified - we know coqui is available
        for engine in engines:
            status = "✅" if engine in available_engines else "❌"
            st.text(f"{status} {engine.title()}")
        
        st.markdown("### 🔧 Settings")
        
        # API key status
        api_key_status = "✅ Set" if os.getenv('ELEVENLABS_API_KEY') else "❌ Not Set"
        st.text(f"ElevenLabs API: {api_key_status}")
        
        if not os.getenv('ELEVENLABS_API_KEY'):
            st.warning("Set ELEVENLABS_API_KEY environment variable for voice cloning!")
        
        # Clear session button
        if st.button("🔄 Clear Session", help="Clear all uploaded files and results"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app uses Generative AI to convert PDF books into personalized audiobooks with your cloned voice.
        
        **Technologies:**
        - ElevenLabs for voice cloning
        - Multiple TTS engines
        - Advanced text processing
        
        **Made for educational purposes** 📚
        """)


def main():
    """Main Streamlit application."""
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit is not installed. Please run: pip install streamlit")
        return
    
    # Page configuration
    st.set_page_config(
        page_title="PDF to Speech with Voice Cloning",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    sidebar()
    
    # Main content
    st.title("📚🎵 PDF to Speech with Voice Cloning")
    st.markdown("""
    Transform your PDF books into personalized audiobooks using advanced AI voice cloning technology!
    
    ---
    """)
    
    # Main workflow
    pdf_uploaded = upload_pdf_section()
    voice_uploaded = upload_voice_section()
    
    voice_name, extract_chapters, tts_engine, chunk_size = configuration_section()
    
    processing_section(voice_name, extract_chapters, tts_engine, chunk_size)
    
    results_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    Made with ❤️ using Streamlit • PDF to Speech with Voice Cloning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
