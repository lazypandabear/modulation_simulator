# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy import signal
from scipy.signal import upfirdn
from scipy.special import erfc

app = Flask(__name__)

def encode_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded

def rrc_filter(beta, span, sps):
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2)
    h[np.isnan(h)] = 0
    h = h / np.sqrt(np.sum(h**2))
    return h

def theoretical_ber(mod_type, snr_db, M):
    snr = 10**(snr_db/10)
    if mod_type == 'BPSK':
        return 0.5 * erfc(np.sqrt(snr))
    elif mod_type == 'QPSK':
        return 0.5 * erfc(np.sqrt(snr/2))
    elif mod_type.startswith('QAM'):
        k = np.log2(M)
        return 4/k * (1 - 1/np.sqrt(M)) * 0.5 * erfc(np.sqrt(3 * k * snr / (M - 1)))
    return 0.5 * erfc(np.sqrt(snr))

def estimate_bandwidth(f, Pxx, threshold_db=-3):
    max_power = np.max(10 * np.log10(Pxx))
    threshold = max_power + threshold_db
    bw_indices = np.where(10 * np.log10(Pxx) >= threshold)[0]
    if bw_indices.size == 0:
        return 0
    bw = f[bw_indices[-1]] - f[bw_indices[0]]
    return round(bw, 2)

def digital_modulate(data, scheme='BPSK', M=2, snr=10, sps=8):
    if scheme == 'BPSK':
        symbols = 2 * data - 1
    elif scheme == 'QPSK':
        data = data.reshape(-1, 2)
        symbols = (2*data[:, 0] - 1) + 1j * (2*data[:, 1] - 1)
    elif scheme.startswith('QAM'):
        bits_per_symbol = int(np.log2(M))
        data = data[:len(data) // bits_per_symbol * bits_per_symbol]
        symbols = []
        for i in range(0, len(data), bits_per_symbol):
            val = int("".join(str(b) for b in data[i:i+bits_per_symbol]), 2)
            x = 2 * (val % int(np.sqrt(M))) - np.sqrt(M) + 1
            y = 2 * (val // int(np.sqrt(M))) - np.sqrt(M) + 1
            symbols.append(complex(x, y))
        symbols = np.array(symbols)
    else:
        symbols = data

    rrc = rrc_filter(beta=0.35, span=10, sps=sps)
    shaped = upfirdn(rrc, symbols, sps)
    noise = np.random.normal(0, 10**(-snr/20), len(shaped))
    return shaped + noise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        mod_type = request.form['mod_type']
        snr = float(request.form.get('snr', 10))
        target_data_rate = float(request.form.get('num_bits', 100)) * 1e6
        M = int(request.form.get('mod_order', 2))
        bitstream = request.form.get('bitstream', '')

        actual_bits_per_symbol = int(np.log2(M)) if M > 1 else 1
        required_bits_per_symbol = actual_bits_per_symbol

        if mod_type == 'QAM':
            standard_mapping = {
                4: 2, 16: 4, 64: 6, 256: 8, 1024: 10, 2048: 11, 4096: 12
            }
            required_bits_per_symbol = standard_mapping.get(M, actual_bits_per_symbol)

        baud_rate = target_data_rate / required_bits_per_symbol
        num_bits = int(baud_rate * actual_bits_per_symbol)

        if bitstream.strip():
            data = np.array([int(b) for b in bitstream.strip() if b in ['0', '1']])
            if len(data) < num_bits:
                data = np.pad(data, (0, num_bits - len(data)), constant_values=0)
            else:
                data = data[:num_bits]
        else:
            data = np.random.randint(0, 2, num_bits)

        signal_out = digital_modulate(data, mod_type, M, snr)

        plt.figure(figsize=(6, 3))
        plt.plot(np.real(signal_out[:100]))
        plt.title("Time Domain Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        time_plot = encode_plot()
        plt.close()

        plt.figure(figsize=(6, 3))
        Fs = 10000
        f, Pxx = signal.welch(np.real(signal_out), fs=Fs, nperseg=1024)
        bw = estimate_bandwidth(f, Pxx)
        plt.semilogy(f - Fs/2, np.fft.fftshift(Pxx))
        plt.title("Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        freq_plot = encode_plot()
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.scatter(np.real(signal_out[::8]), np.imag(signal_out[::8]), s=5)
        plt.title("Constellation Diagram")
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        constellation_plot = encode_plot()
        plt.close()

        ber = theoretical_ber(mod_type, snr, M)

        return jsonify({
            'time_plot': time_plot,
            'freq_plot': freq_plot,
            'constellation_plot': constellation_plot,
            'ber': float(ber),
            'bw': bw,
            'bps': required_bits_per_symbol,
            'bps_actual': actual_bits_per_symbol,
            'data_rate': target_data_rate,
            'baud': round(baud_rate),
            'rsnr': round(10 * np.log10(1 / ber), 2),
            'rChannelBW': bw,
            'throughput': round(target_data_rate / 1e9, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ber_curve', methods=['POST'])
def ber_curve():
    try:
        mod_type = request.form['mod_type']
        M = int(request.form.get('mod_order', 2))
        snrs = np.arange(0, 11)
        bers = [theoretical_ber(mod_type, snr, M) for snr in snrs]

        plt.figure(figsize=(5, 3))
        plt.semilogy(snrs, bers, marker='o')
        plt.title("BER vs. SNR")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate")
        plt.grid(True, which='both')
        ber_plot = encode_plot()
        plt.close()

        return jsonify({'ber_curve': ber_plot})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)