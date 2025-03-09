import time
import os
import tempfile
import threading
from googletrans import Translator
from gtts import gTTS

class SequenceProcessor:
    def __init__(self, settings):
        self.settings = settings
        self.sequence = []
        self.last_spoken_time = 0
        self.MIN_AUDIO_INTERVAL = 2.0
        self.audio_thread = None
        self.translator = Translator()
        self.translated_text = ""
        self.max_sequence_length = 30  # Limit sequence length to prevent memory issues
        
    def add_to_sequence(self, text):
        """Add detected text to the sequence"""
        if len(self.sequence) >= self.max_sequence_length:
            # Remove oldest item if max length reached
            self.sequence.pop(0)
        
        self.sequence.append(text)
        # Update translation whenever sequence changes
        self._update_translation()
        
    def get_sequence(self):
        """Get the current sequence as a string"""
        return " ".join(self.sequence)
        
    def get_translated_text(self):
        """Get the translated text"""
        return self.translated_text
        
    def _update_translation(self):
        """Update the translated text based on current sequence and language settings"""
        if not self.sequence:
            self.translated_text = ""
            return
            
        sequence_text = self.get_sequence()
        
        try:
            if self.settings['language'] != 'en':
                self.translated_text = self.translator.translate(
                    sequence_text, dest=self.settings['language']).text
            else:
                self.translated_text = sequence_text
        except Exception as e:
            print(f"Translation error: {e}")
            self.translated_text = sequence_text  # Use original text if translation fails
    
    def clear_sequence(self):
        """Clear the current sequence"""
        self.sequence = []
        self.translated_text = ""
        
    def speak_sequence(self):
        """Speak the current sequence in the selected language"""
        if not self.sequence:
            return
            
        current_time = time.time()
        sequence_text = self.get_sequence()
        
        # Check if enough time has passed and previous audio has finished
        if (current_time - self.last_spoken_time >= self.MIN_AUDIO_INTERVAL and 
            self.settings['audio_enabled'] and 
            (self.audio_thread is None or not self.audio_thread.is_alive())):
            
            # Start new audio thread
            self.audio_thread = self._speak_text(sequence_text, self.settings['language'])
            if self.audio_thread is not None:
                self.last_spoken_time = current_time
    
    def _speak_text(self, text, lang_code):
        """Convert text to speech"""
        try:
            if lang_code != 'en':
                # Use pre-translated text if available
                translated = self.translated_text if self.translated_text else self.translator.translate(
                    text, dest=lang_code).text
            else:
                translated = text
            
            # Create a temporary file with proper permissions
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                
            try:
                tts = gTTS(text=translated, lang=lang_code)
                tts.save(temp_path)
                
                # Use playsound in a separate thread to avoid blocking
                from playsound import playsound
                def play_and_cleanup():
                    try:
                        playsound(temp_path)
                    finally:
                        # Clean up the temporary file after playing
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                
                thread = threading.Thread(target=play_and_cleanup, daemon=True)
                thread.start()
                return thread
                
            except Exception as e:
                print(f"Error in audio playback: {e}")
                # Clean up the temporary file if speech synthesis fails
                try:
                    os.remove(temp_path)
                except:
                    pass
                
        except Exception as e:
            print(f"Error in speech synthesis: {e}")
        return None 

    def speak_final_sequence(self):
        """Speak the current sequence and ensure it completes even after window closure"""
        if not self.sequence:
            return
        
        sequence_text = self.get_sequence()
        
        try:
            if self.settings['language'] != 'en':
                # Use pre-translated text if available
                translated = self.translated_text if self.translated_text else self.translator.translate(
                    sequence_text, dest=self.settings['language']).text
            else:
                translated = sequence_text
            
            # Create a temporary file with proper permissions
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Generate speech
            tts = gTTS(text=translated, lang=self.settings['language'])
            tts.save(temp_path)
            
            # Play without thread to ensure it completes
            from playsound import playsound
            playsound(temp_path)
            
            # Clean up the temporary file after playing
            try:
                os.remove(temp_path)
            except:
                pass
            
        except Exception as e:
            print(f"Error in final speech synthesis: {e}") 