#!/usr/bin/env python3
"""
Simplified Streamlit Interface - Two Step Process
"""

import streamlit as st
import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import PDFToSpeechApp

def initialize_app():
    """Initialize the application in session state."""
    if 'app' not in st.session_state:
        st.session_state.app = PDFToSpeechApp()

def step1_section():
    """STEP 1: PDF to Audio conversion."""
    st.header("📖 STEP 1: PDF to Audio")
    st.markdown("**Upload your PDF and get an audio file with default voice**")
    
    # PDF upload
    pdf_file = st.file_uploader(
        "Upload your PDF document",
        type=['pdf'],
        help="Upload the PDF book you want to convert to audio"
    )
    
    if pdf_file:
        st.success(f"✅ PDF uploaded: {pdf_file.name}")
        
        # Save PDF temporarily
        temp_dir = Path(tempfile.gettempdir()) / "pdf_to_speech"
        temp_dir.mkdir(exist_ok=True)
        pdf_path = temp_dir / f"uploaded_{pdf_file.name}"
        
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Convert to audio button
        if st.button("🎵 Convert PDF to Audio", type="primary"):
            with st.spinner("Converting PDF to audio..."):
                result = st.session_state.app.step1_pdf_to_audio(str(pdf_path))
                
                if result["success"]:
                    st.success("✅ " + result["message"])
                    
                    # Show download button
                    audio_file_path = Path(result["audio_file"])
                    if audio_file_path.exists():
                        with open(audio_file_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Audio File",
                                data=f.read(),
                                file_name=f"{pdf_file.name.replace('.pdf', '')}_audio.wav",
                                mime="audio/wav"
                            )
                        
                        st.session_state.step1_completed = True
                        st.session_state.step1_audio_file = str(audio_file_path)
                        
                        st.info("💡 **Next**: Download this audio file, then proceed to Step 2 for voice cloning!")
                    else:
                        st.error("Audio file was not created properly")
                else:
                    st.error("❌ " + result["message"])

def step2_section():
    """STEP 2: Voice cloning."""
    st.header("🎤 STEP 2: Voice Cloning")
    st.markdown("**Upload the audio from Step 1 + your voice sample to get personalized audiobook**")
    
    # Audio file upload (from Step 1)
    original_audio = st.file_uploader(
        "Upload the audio file from Step 1",
        type=['wav', 'mp3'],
        help="Upload the audio file you downloaded from Step 1"
    )
    
    # Voice sample upload
    voice_sample = st.file_uploader(
        "Upload your voice sample",
        type=['wav', 'mp3', 'ogg', 'opus', 'm4a'],
        help="Upload your voice recording (WhatsApp .ogg files supported!)"
    )
    
    if original_audio and voice_sample:
        st.success(f"✅ Files uploaded:")
        st.write(f"   • Original audio: {original_audio.name}")
        st.write(f"   • Voice sample: {voice_sample.name}")
        
        # Save files temporarily
        temp_dir = Path(tempfile.gettempdir()) / "pdf_to_speech"
        temp_dir.mkdir(exist_ok=True)
        
        original_path = temp_dir / f"original_{original_audio.name}"
        voice_path = temp_dir / f"voice_{voice_sample.name}"
        
        with open(original_path, "wb") as f:
            f.write(original_audio.getbuffer())
        
        with open(voice_path, "wb") as f:
            f.write(voice_sample.getbuffer())
        
        # Output name
        output_name = st.text_input("Name for your cloned audiobook", value="my_audiobook")
        
        # Clone voice button
        if st.button("🎭 Clone Voice & Generate Audiobook", type="primary"):
            with st.spinner("Cloning voice and regenerating audio..."):
                result = st.session_state.app.step2_clone_voice(
                    original_audio_path=str(original_path),
                    voice_sample_path=str(voice_path),
                    output_name=output_name
                )
                
                if result["success"]:
                    st.success("✅ " + result["message"])
                    
                    # Show download button for cloned audio
                    cloned_file_path = Path(result["cloned_audio_file"])
                    if cloned_file_path.exists():
                        with open(cloned_file_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Cloned Audiobook",
                                data=f.read(),
                                file_name=f"{output_name}_cloned.wav",
                                mime="audio/wav"
                            )
                        
                        st.balloons()
                        st.success("🎉 Your personalized audiobook is ready!")
                    else:
                        st.error("Cloned audio file was not created properly")
                else:
                    st.error("❌ " + result["message"])

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="PDF to Speech with Voice Cloning",
        page_icon="🎵",
        layout="wide"
    )
    
    # Initialize app
    initialize_app()
    
    # Title
    st.title("🎵 PDF to Speech with Voice Cloning")
    st.markdown("**Simple Two-Step Process**")
    
    # Instructions
    with st.expander("📋 How it Works", expanded=True):
        st.markdown("""
        ### 🎯 Simple Two-Step Process:
        
        **STEP 1: PDF → Audio**
        1. Upload your PDF document
        2. Convert to audio with default voice
        3. Download the audio file
        
        **STEP 2: Voice Cloning**
        1. Upload the downloaded audio file from Step 1
        2. Upload your voice sample (WhatsApp .ogg files work!)
        3. Get your personalized audiobook with your voice
        
        ### 📱 WhatsApp Voice Messages
        - ✅ **Supported**: Upload .ogg files directly from WhatsApp
        - ✅ **No conversion needed**: The app handles .ogg files automatically
        """)
    
    # Create two columns for the two steps
    col1, col2 = st.columns(2)
    
    with col1:
        step1_section()
    
    with col2:
        step2_section()
    
    # Footer
    st.markdown("---")
    st.markdown("🔧 **Status**: Ready to process PDFs and clone voices!")

if __name__ == "__main__":
    main()
