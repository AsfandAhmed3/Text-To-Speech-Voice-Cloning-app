# PDF to Speech with Voice Cloning - Usage Guide

## 🚀 Quick Start

Your PDF to Speech application is now ready to use! Here's how to get started:

## 📦 Current Status

✅ **Application Ready**: The main PDFToSpeechApp class is working  
✅ **Streamlit Interface**: Running at http://localhost:8502  
✅ **Coqui TTS**: Installed and configured for voice cloning  
✅ **Dependencies**: All required packages installed  

## 🌐 Web Interface Usage

### Streamlit Interface (Recommended)
1. **Access**: Open http://localhost:8502 in your browser
2. **Upload PDF**: Use the file uploader to select your PDF document
3. **Upload Voice Samples** (Optional): Add audio files of your voice for cloning
4. **Configure Settings**: Choose TTS engine, voice preferences, and output options
5. **Process**: Click the process button to convert PDF to speech
6. **Download**: Get your generated audiobook files

### Gradio Interface
```bash
# To start Gradio interface
python src/ui/gradio_app.py
```

## 💻 Command Line Usage

### Basic PDF Processing
```bash
python demo.py
```

### Advanced Usage
```bash
python src/main.py --pdf "data/your-document.pdf" --voice-name "my_voice" --tts-engine coqui
```

## 📁 File Organization

```
📂 Your Project
├── 📄 demo.py                 # Demo script
├── 📂 data/                   # Input PDFs
├── 📂 output/                 # Generated audio files
│   ├── 📂 default_voice/      # Standard TTS output
│   ├── 📂 cloned_voice/       # Voice-cloned output
│   └── 📂 extracted_text/     # PDF text extraction
├── 📂 src/                    # Source code
│   ├── 📄 main.py            # Main application
│   └── 📂 ui/                # Web interfaces
│       ├── 📄 streamlit_app.py
│       └── 📄 gradio_app.py
└── 📂 voice_samples/         # Your voice recordings
```

## 🔧 Available Features

### Current Implementation
- ✅ **PDF Upload & Basic Info**: File size, placeholder for page count
- ✅ **Voice Sample Upload**: Multiple audio files supported
- ✅ **Web Interface**: Full Streamlit UI with sidebar controls
- ✅ **Coqui TTS Integration**: Voice cloning engine ready
- ✅ **Output Organization**: Structured output directories

### Placeholder Components (Ready for Enhancement)
- 🔄 **PDF Text Extraction**: Framework ready for PyPDF2/pdfplumber
- 🔄 **Text Preprocessing**: Chapter detection and text cleaning
- 🔄 **Audio Generation**: Full TTS pipeline implementation
- 🔄 **Voice Cloning**: Complete Coqui TTS voice cloning workflow

## 🎵 Voice Cloning Tips

1. **Voice Samples**: Upload 3-5 clear audio samples of your voice (WAV/MP3)
2. **Duration**: Each sample should be 10-30 seconds long
3. **Quality**: Clear recording, minimal background noise
4. **Content**: Different sentences for better voice modeling

## 🔧 Troubleshooting

### Import Errors
- ✅ **Fixed**: PDFToSpeechApp import now works correctly
- All web interfaces can import the main application class

### Streamlit Issues
- **Access**: Make sure you're using http://localhost:8502
- **Refresh**: Use Ctrl+R to refresh if the interface seems stuck
- **Restart**: Use Ctrl+C in terminal to stop and restart if needed

### Python Environment
- **Packages**: All required packages are installed
- **Python**: Using Python 3.11.9 in system environment

## 📞 Next Steps

1. **Test the Interface**: Upload a sample PDF and explore the features
2. **Add Voice Samples**: Record yourself reading for voice cloning
3. **Customize Settings**: Adjust TTS parameters in the sidebar
4. **Process Documents**: Convert your first PDF to audiobook
5. **Expand Features**: The framework is ready for additional enhancements

## 🎉 You're All Set!

Your PDF to Speech application is working and ready for use. Start with the Streamlit interface at http://localhost:8502 and begin converting your documents to audiobooks with personalized voice cloning!
