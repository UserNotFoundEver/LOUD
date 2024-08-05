from pydub import AudioSegment
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import os
import io
import base64
from threading import Timer

app = Flask(__name__)

def read_audio(file_path):
    return AudioSegment.from_file(file_path)

def detect_loudness(audio):
    samples = np.array(audio.get_array_of_samples())
    rms = np.sqrt(np.mean(samples**2))
    return rms

def detect_noise_type(samples, rate):
    # FFT and frequency analysis for noise detection
    n = len(samples)
    freqs = np.fft.fftfreq(n, 1/rate)
    fft_values = np.abs(fft(samples))
    
    # Simple detection based on frequency content
    if np.mean(fft_values[(freqs > 10) & (freqs < 100)]) > np.mean(fft_values[(freqs > 1000) & (freqs < 2000)]):
        return "Pink Noise"
    elif np.mean(fft_values[(freqs > 1000) & (freqs < 2000)]) > np.mean(fft_values[(freqs > 10) & (freqs < 100)]):
        return "White Noise"
    else:
        return "Unknown Noise"

def plot_waveform(samples, rate):
    time = np.linspace(0., len(samples) / rate, len(samples))
    plt.figure(figsize=(10, 4))
    plt.plot(time, samples, label="Waveform")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file[]')
        results = []
        for file in files:
            audio = read_audio(file)
            samples = np.array(audio.get_array_of_samples())
            rate = audio.frame_rate
            loudness = detect_loudness(audio)
            noise_type = detect_noise_type(samples, rate)

            # Plot waveform
            img = io.BytesIO()
            plot_waveform(samples, rate)
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            results.append({
                'filename': file.filename,
                'loudness': loudness,
                'noise_type': noise_type,
                'plot_url': plot_url
            })

        return render_template('results.html', results=results)

    return '''
    <!doctype html>
    <title>Upload Audio Files</title>
    <h1>Upload up to 10 MP3 or WAV files</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file[]" multiple>
      <input type="submit" value="Upload">
    </form>
    '''

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

def run_app():
    Timer(600, shutdown_server).start()  # Shut down the server after 10 minutes
    app.run(debug=True)

if __name__ == '__main__':
    run_app()
