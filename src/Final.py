import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, simpledialog, Listbox, Scrollbar
import os
from elevenlabs import ElevenLabs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pygame import mixer
from pydub import AudioSegment
from io import BytesIO

# Initialize ElevenLabs client
client = ElevenLabs(api_key='sk_6d69377342f29ec53a60f93368ecfb811e8e2edf668589ef')  # Replace with your actual API key

# Define the path to the saved model
save_directory = "C:/Users/riadm/Desktop/FinalProjesiIngilizce/emotion_model"

# Load the saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSequenceClassification.from_pretrained(save_directory)

# Correct emotion labels
emotion_labels = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]

class EmotionDubbingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Dubbing Application")
        self.root.geometry("700x600")
        self.style = ttk.Style("superhero")  # Use a modern theme

        # Title Label
        self.title_label = ttk.Label(root, text="Emotion Dubbing Application", font=("Helvetica", 20, "bold"), anchor=CENTER)
        self.title_label.pack(pady=20)

        # Text input frame
        input_frame = ttk.Frame(root, padding=10)
        input_frame.pack(fill=X, padx=20)

        self.text_label = ttk.Label(input_frame, text="Enter Text:", font=("Helvetica", 12, "bold"))
        self.text_label.pack(side=LEFT)

        self.text_entry = ttk.Text(input_frame, height=5, width=60, font=("Helvetica", 12), wrap="word", bd=2, relief="groove")
        self.text_entry.pack(pady=10)

        # Voice selection frame
        voice_frame = ttk.Frame(root, padding=10)
        voice_frame.pack(fill=X, padx=20)

        self.voice_label = ttk.Label(voice_frame, text="Select Voice:", font=("Helvetica", 12, "bold"))
        self.voice_label.pack(side=LEFT)

        # ComboBox for selecting voice ID
        self.voice_combobox = ttk.Combobox(voice_frame, values=["nPczCjzI2devNBz1zQrb", "g5CIjZEefAph4nQFvHAz", "9BWtsMINqrJLrRacOk9x"], state="readonly", font=("Helvetica", 12))
        self.voice_combobox.set("nPczCjzI2devNBz1zQrb")  # Default voice ID
        self.voice_combobox.pack(pady=10, side=LEFT)

        # Buttons frame
        button_frame = ttk.Frame(root, padding=10)
        button_frame.pack(fill=X, pady=10)

        self.convert_button = ttk.Button(button_frame, text="Convert", bootstyle=PRIMARY, command=self.convert_text)
        self.convert_button.pack(side=LEFT, padx=10)

        self.play_button = ttk.Button(button_frame, text="Play", bootstyle=SUCCESS, command=self.play_audio, state=DISABLED)
        self.play_button.pack(side=LEFT, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop", bootstyle=DANGER, command=self.stop_audio, state=DISABLED)
        self.stop_button.pack(side=LEFT, padx=10)

        # Status label
        self.status_label = ttk.Label(root, text="Status: Ready", font=("Helvetica", 12), anchor=CENTER, bootstyle=INFO)
        self.status_label.pack(pady=10)

        # Audio file label
        self.audio_file_label = ttk.Label(root, text="WAV File: Not created yet", font=("Helvetica", 10), anchor=CENTER)
        self.audio_file_label.pack(pady=10)

        # Emotion listbox
        self.emotion_label = ttk.Label(root, text="Predicted Emotions and Confidence:", font=("Helvetica", 12, "bold"))
        self.emotion_label.pack(pady=10)

        # Frame for listbox and scrollbar
        listbox_frame = ttk.Frame(root)
        listbox_frame.pack(pady=10)

        # Listbox
        self.emotion_listbox = Listbox(
            listbox_frame,
            height=10,
            width=80,
            font=("Helvetica", 12),
            selectmode="browse",
            bd=2,
            relief="groove"
        )
        self.emotion_listbox.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = Scrollbar(listbox_frame, orient="vertical", command=self.emotion_listbox.yview)
        scrollbar.pack(side="right", fill="y")

        # Attach scrollbar to listbox
        self.emotion_listbox.config(yscrollcommand=scrollbar.set)

        self.emotion_listbox.pack(pady=10)  

        # Audio file path and emotion results
        self.audio_file = None
        self.emotion_results = []

    def convert_text(self):
        text = self.text_entry.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showerror("Error", "Please enter text!")
            return
        
        # Ask for file name
        file_name = simpledialog.askstring("File Name", "Enter the name of the audio file:", parent=self.root)
        
        if not file_name:
            messagebox.showerror("Error", "Please enter a valid file name!")
            return
        
        # Save path for the audio file
        save_path = os.path.join(os.getcwd(), f"{file_name}.wav")
        if os.path.exists(save_path):
            messagebox.showerror("Error", f"A file named {file_name}.wav already exists. Please choose a different name.")
            return
        
        self.status_label.config(text="Status: Processing...", bootstyle=WARNING)
        
        try:
            # Analyze text and generate audio
            self.emotion_results = self.analyze_text_and_generate_audio(text, save_path)
            self.audio_file = save_path
            self.audio_file_label.config(text=f"WAV File: {self.audio_file}")
            self.status_label.config(text="Status: Conversion Complete", bootstyle=SUCCESS)
            self.play_button.config(state=NORMAL)
            
            # Display emotion results in the listbox
            self.emotion_listbox.delete(0, END)
            for sentence, emotion, confidence in self.emotion_results:
                self.emotion_listbox.insert(END, f"Sentence: {sentence} | Emotion: {emotion} | Confidence: {confidence:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_label.config(text="Status: Error", bootstyle=DANGER)

    def analyze_text_and_generate_audio(self, text, save_path):
        sentences = text.split(". ")
        results = []
        combined_text = ""
        for sentence in sentences:
            if sentence.strip():
                inputs = tokenizer(sentence.strip(), return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_emotion_index = torch.argmax(probs).item()
                emotion = emotion_labels[top_emotion_index]
                confidence = probs[0, top_emotion_index].item() * 100
                results.append((sentence.strip(), emotion, confidence))
                combined_text += sentence.strip() + " "

        # Get selected voice ID from ComboBox
        selected_voice = self.voice_combobox.get()

        # Generate TTS audio stream with selected voice
        audio_stream = client.text_to_speech.convert_as_stream(
            text=combined_text,
            voice_id=selected_voice,
            model_id="eleven_multilingual_v2"
        )

        # Save audio using pydub
        audio = AudioSegment.from_file(BytesIO(b"".join(audio_stream)), format="mp3")
        audio.export(save_path, format="wav")

        return results

    def play_audio(self):
        if self.audio_file:
            mixer.init()
            mixer.music.load(self.audio_file)
            mixer.music.play()
            self.stop_button.config(state=NORMAL)
            self.play_button.config(state=DISABLED)

    def stop_audio(self):
        mixer.music.stop()
        self.stop_button.config(state=DISABLED)
        self.play_button.config(state=NORMAL)

# Run the GUI
if __name__ == "__main__":
    root = ttk.Window(themename="superhero")  # You can experiment with other themes like "darkly", "solar", etc.
    app = EmotionDubbingApp(root)
    root.mainloop()
