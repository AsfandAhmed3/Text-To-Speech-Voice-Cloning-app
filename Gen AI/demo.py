#!/usr/bin/env python3
"""
Demo script for PDF to Speech with Voice Cloning application

This script demonstrates how to use the PDF to Speech application
both programmatically and through the web interfaces.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import PDFToSpeechApp

def main():
    """Demonstrate the PDF to Speech application usage."""
    
    print("🎵 PDF to Speech with Voice Cloning - Demo")
    print("=" * 50)
    
    # Initialize the application
    print("\n📝 Initializing application...")
    app = PDFToSpeechApp(
        tts_engine="coqui",
        voice_service="coqui",
        output_dir="output"
    )
    print("✅ Application initialized successfully!")
    
    # Show available sample PDF
    sample_pdf = Path("data/max-tegmark-life-30-being-human-in-the-age-of-artificial-intelligence-alfred-a-knopf-2017-aTvn.pdf")
    
    if sample_pdf.exists():
        print(f"\n📄 Sample PDF found: {sample_pdf.name}")
        
        # Demonstrate workflow (placeholder)
        print("\n🔄 Running workflow demo...")
        results = app.process_full_workflow(
            pdf_path=str(sample_pdf),
            voice_sample_paths=None,  # No voice samples for basic demo
            voice_name="demo_voice"
        )
        
        if results["success"]:
            print("✅ Demo workflow completed successfully!")
            print(f"📊 Results: {results}")
        else:
            print("❌ Demo workflow failed!")
            for error in results["errors"]:
                print(f"   Error: {error}")
    else:
        print(f"\n⚠️  Sample PDF not found at: {sample_pdf}")
        print("   You can add your own PDF to the 'data' folder")
    
    print("\n🌐 Web Interfaces:")
    print("   • Streamlit: Currently running at http://localhost:8502")
    print("   • Gradio: Run 'python src/ui/gradio_app.py' to start")
    
    print("\n📚 Available Features:")
    print("   ✅ PDF text extraction (placeholder)")
    print("   ✅ Voice cloning with Coqui TTS (ready)")  
    print("   ✅ Text-to-speech conversion (placeholder)")
    print("   ✅ Web interface (Streamlit & Gradio)")
    
    print("\n🚀 Next Steps:")
    print("   1. Upload a PDF using the web interface")
    print("   2. Optionally upload voice samples for cloning")
    print("   3. Configure your preferences")
    print("   4. Generate your audiobook!")
    
    print("\nDemo completed! 🎉")

if __name__ == "__main__":
    main()
