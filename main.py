from tkinter import *
from PIL import ImageTk, Image
import tkinter.messagebox as tkMessageBox
import ctypes
import os
import json
from tkinter import ttk
import threading
import time

# Import custom modules
from sequence_processor import SequenceProcessor
from alphabet_detector import AlphabetDetector
from word_detector import WordDetector

# Define supported languages
SUPPORTED_LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Malayalam': 'ml',
    'Kannada': 'kn',
    'Bengali': 'bn',
    'Marathi': 'mr'
}

# Load settings from file or use defaults
try:
    with open('settings.json', 'r') as f:
        SETTINGS = json.load(f)
except:
    SETTINGS = {
        'language': 'en',
        'audio_enabled': True,
        'frame_delay': 15
    }

# Initialize sequence processor
sequence_processor = SequenceProcessor(SETTINGS)

# Setup main window
directory = "./"
home = Tk()
home.title("Sign To Text and Speech")
img = Image.open(directory+"/assets/home.jpeg")
img = ImageTk.PhotoImage(img)
panel = Label(home, image=img)
panel.pack(side="top", fill="both", expand="yes")
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0]//2-600)
b= str(lt[1]//2-360)
home.geometry("1200x720+"+a+"+"+b)
home.resizable(0,0)


def Exit():
    global home
    result = tkMessageBox.askquestion(
        "Sign To Text and Speech", 'Are you sure you want to exit?', icon="warning")
    if result == 'yes':
        home.destroy()
        exit()

def digitalpha():
    # Clear the sequence before starting a new detection
    sequence_processor.clear_sequence()
    detector = AlphabetDetector(sequence_processor, SETTINGS)
    detector.run()
    # After window is closed, speak the final sequence if needed
    if sequence_processor.sequence and SETTINGS['audio_enabled']:
        threading.Thread(target=sequence_processor.speak_final_sequence, daemon=True).start()
    
def yolo():
    # Clear the sequence before starting a new detection
    sequence_processor.clear_sequence()
    detector = WordDetector(sequence_processor, SETTINGS)
    detector.run()
    # After window is closed, speak the final sequence if needed
    if sequence_processor.sequence and SETTINGS['audio_enabled']:
        threading.Thread(target=sequence_processor.speak_final_sequence, daemon=True).start()

def settings():
    settings_window = Toplevel(home)
    settings_window.title("Settings")
    settings_window.geometry("400x300")
    settings_window.resizable(False, False)

    # Language Selection
    Label(settings_window, text="Select Language:", font=('Arial', 12)).pack(pady=10)
    
    # Get the language name from the code
    current_lang_name = "English"  # Default
    for name, code in SUPPORTED_LANGUAGES.items():
        if code == SETTINGS['language']:
            current_lang_name = name
            break
            
    language_var = StringVar(value=current_lang_name)
    language_combo = ttk.Combobox(settings_window, textvariable=language_var, 
                                 values=list(SUPPORTED_LANGUAGES.keys()))
    language_combo.pack(pady=5)

    # Audio Toggle
    audio_var = BooleanVar(value=SETTINGS['audio_enabled'])
    audio_check = Checkbutton(settings_window, text="Enable Audio", 
                             variable=audio_var, font=('Arial', 12))
    audio_check.pack(pady=10)

    def save_settings():
        SETTINGS['language'] = SUPPORTED_LANGUAGES[language_combo.get()]
        SETTINGS['audio_enabled'] = audio_var.get()
        
        # Save settings to file
        with open('settings.json', 'w') as f:
            json.dump(SETTINGS, f)
        
        # Update sequence processor settings
        sequence_processor.settings = SETTINGS
        
        settings_window.destroy()
        tkMessageBox.showinfo("Success", "Settings saved successfully!")

    Button(settings_window, text="Save", command=save_settings, 
           font=('Arial', 12)).pack(pady=20)

def about():
    tkMessageBox.showinfo(
        'About Us', """Sign to Text and Speech is an innovative application that bridges communication gaps by translating sign language into text and speech in real-time. It supports both alphabets and common words/phrases, making it a versatile tool for sign language users.

Features:
- Real-time sign language detection
- Support for both alphabets and common phrases
- Multi-language speech output
- Adjustable settings for personalization
- User-friendly interface
- Sequence processing to build sentences
- Text translation in multiple Indian languages

This application aims to make communication more accessible and inclusive for the deaf and hard of hearing community.""")
       
photo = Image.open(directory+"assets/1.jpeg")
img3 = ImageTk.PhotoImage(photo)
b2=Button(home, highlightthickness = 0, bd = 0,activebackground="#e4e4e4", image = img3,command=digitalpha)
b2.place(x=69,y=221)

photo = Image.open(directory+"assets/2.jpeg")
img2 = ImageTk.PhotoImage(photo)
b1=Button(home, highlightthickness = 0, bd = 0,activebackground="white", image = img2,command=yolo)
b1.place(x=69,y=330)

photo = Image.open(directory+"assets/3.jpeg")
img4 = ImageTk.PhotoImage(photo)
b1=Button(home, highlightthickness = 0, bd = 0,activebackground="white", image = img4,command=settings)
b1.place(x=71,y=438)

photo = Image.open(directory+"assets/4.png")
img5 = ImageTk.PhotoImage(photo)
b1=Button(home, highlightthickness = 0, bd = 0,activebackground="white", image = img5,command=about)
b1.place(x=71,y=531)

photo = Image.open(directory+"assets/5.png")
img6 = ImageTk.PhotoImage(photo)
b1=Button(home, highlightthickness = 0, bd = 0,activebackground="white", image = img6,command=Exit)
b1.place(x=385,y=531)

# Define save_sequence function before using it
def save_sequence():
    """Save current sequence to a file"""
    if not sequence_processor.sequence:
        tkMessageBox.showinfo("Information", "No sequence to save.")
        return
        
    sequence_text = sequence_processor.get_sequence()
    translated_text = sequence_processor.get_translated_text()
    
    # Create a directory for saved sequences if it doesn't exist
    os.makedirs("saved_sequences", exist_ok=True)
    
    # Generate a timestamp for the filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"saved_sequences/sequence_{timestamp}.txt"
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Original: {sequence_text}\n")
        f.write(f"Translated: {translated_text}\n")
    
    tkMessageBox.showinfo("Success", f"Sequence saved to {filename}")

# Now create the button after the function is defined
save_button = Button(home, text="Save Sequence", command=save_sequence, 
                    font=('Arial', 12), bg="#4CAF50", fg="white", 
                    activebackground="#45a049", activeforeground="white",
                    padx=10, pady=5)
save_button.place(x=225, y=620)  # Adjust position as needed

home.mainloop()
