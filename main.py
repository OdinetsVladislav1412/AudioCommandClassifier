import tkinter as tk
from audio_app import AudioRecorderApp

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Классификатор речевых команд")
    
    app = AudioRecorderApp(root)
    root.mainloop()