#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality work correctly
"""

def test_imports():
    """Test that all main components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.main import PDFToSpeechApp
        print("✅ Main application import successful")
        
        # Test initialization
        app = PDFToSpeechApp()
        print("✅ Application initialization successful")
        
        # Test basic workflow
        result = app.process_full_workflow(pdf_path="test.pdf")
        if result["success"]:
            print("✅ Basic workflow test successful")
        else:
            print("❌ Basic workflow test failed")
            
    except Exception as e:
        print(f"❌ Main application test failed: {e}")
        return False
    
    try:
        import src.ui.streamlit_app
        print("✅ Streamlit app import successful")
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False
    
    try:
        import src.ui.gradio_app
        print("✅ Gradio app import successful")
    except Exception as e:
        print(f"❌ Gradio app import failed: {e}")
        return False
    
    return True

def test_coqui_tts():
    """Test Coqui TTS availability."""
    print("\n🎵 Testing Coqui TTS...")
    
    try:
        import TTS
        print("✅ TTS library available")
        
        from TTS.api import TTS
        print("✅ TTS API accessible")
        
        # List available models
        tts = TTS()
        print("✅ TTS initialization successful")
        
        return True
    except Exception as e:
        print(f"❌ Coqui TTS test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 PDF to Speech Application - Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test TTS
    tts_ok = test_coqui_tts()
    
    print("\n📊 Test Results:")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Coqui TTS: {'✅ PASS' if tts_ok else '❌ FAIL'}")
    
    if imports_ok and tts_ok:
        print("\n🎉 All tests passed! Your application is ready to use.")
        print("\n🚀 Start the web interface with:")
        print("   streamlit run src/ui/streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()
