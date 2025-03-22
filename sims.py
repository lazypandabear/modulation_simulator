# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy import signal

app = Flask(__name__)

# Helper: Encode plots to base64
def encode_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded

# Helper: Generate simple digital modulation (BPSK, QPSK, etc.)
def digital_modulate(data, scheme='BPSK', snr=10):
    if scheme == 'BPSK':
        symbols = 2 * data - 1
        noise = np.random.normal(0, 10**(-snr/20), len(symbols))
        return symbols + noise
    elif scheme == 'QPSK':
        data = data.reshape(-1, 2)
        symbols = (2*data[:, 0] - 1) + 1j * (2*data[:, 1] - 1)
        noise = np.random.normal(0, 10**(-snr/20), len(symbols)) + 1j * np.random.normal(0, 10**(-snr/20), len(symbols))
        return symbols + noise
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        mod_type = request.form['mod_type']
        snr = float(request.form.get('snr', 10))
        num_bits = int(request.form.get('num_bits', 100))

        data = np.random.randint(0, 2, num_bits)

        modulated = digital_modulate(data, mod_type, snr)
        
        # Time domain
        plt.figure()
        plt.plot(np.real(modulated))
        plt.title("Time Domain Signal")
        time_plot = encode_plot()

        # Frequency domain
        plt.figure()
        plt.magnitude_spectrum(np.real(modulated), Fs=1, scale='dB')
        plt.title("Frequency Domain")
        freq_plot = encode_plot()

        # Constellation
        plt.figure()
        plt.scatter(np.real(modulated), np.imag(modulated))
        plt.title("Constellation Diagram")
        constellation_plot = encode_plot()

        # BER (dummy for now)
        ber = 0.5 * np.exp(-snr/10)
        
        return jsonify({
            'time_plot': time_plot,
            'freq_plot': freq_plot,
            'constellation_plot': constellation_plot,
            'ber': ber,
            'bw': 1,
            'bps': 1 if mod_type=='BPSK' else 2,
            'data_rate': 1 * (1 if mod_type=='BPSK' else 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
