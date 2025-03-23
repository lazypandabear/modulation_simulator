from flask import Flask, render_template, request
import math
import random

app = Flask(__name__)

STANDARD_VALUES = {
    "QAM": {
        2:  {"snr_req": 8,  "bw_req": 10,  "bits_per_symbol": 1,  "baud_rate": 1e6, "throughput": 1e6},
        4:  {"snr_req": 12, "bw_req": 20,  "bits_per_symbol": 2,  "baud_rate": 1e6, "throughput": 2e6},
        8:  {"snr_req": 16, "bw_req": 30,  "bits_per_symbol": 3,  "baud_rate": 1e6, "throughput": 3e6},
        16: {"snr_req": 20, "bw_req": 40,  "bits_per_symbol": 4,  "baud_rate": 1e6, "throughput": 4e6},
        32: {"snr_req": 24, "bw_req": 50,  "bits_per_symbol": 5,  "baud_rate": 1e6, "throughput": 5e6}
    },
    "PSK": {
        2:  {"snr_req": 9,  "bw_req": 15,  "bits_per_symbol": 1,  "baud_rate": 1e6, "throughput": 1e6},
        4:  {"snr_req": 13, "bw_req": 25,  "bits_per_symbol": 2,  "baud_rate": 1e6, "throughput": 2e6},
        8:  {"snr_req": 17, "bw_req": 35,  "bits_per_symbol": 3,  "baud_rate": 1e6, "throughput": 3e6}
    },
    "FSK": {
        2:  {"snr_req": 7,  "bw_req": 12,  "bits_per_symbol": 1,  "baud_rate": 1e6, "throughput": 1e6},
        4:  {"snr_req": 11, "bw_req": 22,  "bits_per_symbol": 2,  "baud_rate": 1e6, "throughput": 2e6}
    },
    "ASK": {
        2:  {"snr_req": 10, "bw_req": 18,  "bits_per_symbol": 1,  "baud_rate": 1e6, "throughput": 1e6},
        4:  {"snr_req": 14, "bw_req": 28,  "bits_per_symbol": 2,  "baud_rate": 1e6, "throughput": 2e6}
    },
    "CSS": {
        2:  {"snr_req": 6,  "bw_req": 9,   "bits_per_symbol": 1,  "baud_rate": 1e6, "throughput": 1e6},
        4:  {"snr_req": 10, "bw_req": 19,  "bits_per_symbol": 2,  "baud_rate": 1e6, "throughput": 2e6}
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    mod_type = "QAM"
    mod_order = 2
    snr = 10
    target_data_rate = 1.0
    sample_data = "0"
    auto_compute = False

    calc_values = {}
    standard_values = {}
    time_domain_data = []
    time_values = []
    index_labels = []
    freq_data = []
    ber_snr_data = []
    constellation_data = []
    noise_data = []

    if request.method == 'POST':
        mod_type = request.form.get("mod_type", "QAM")
        mod_order = int(request.form.get("mod_order", 2))
        snr = float(request.form.get("snr", 10))
        target_data_rate = float(request.form.get("target_data_rate", 1.0))
        sample_data = request.form.get("sample_data", "")
        auto_compute = request.form.get("auto_compute") == "on"

        bits_per_symbol = int(math.log2(mod_order))

        # Validate input: length must be a multiple of bits_per_symbol and only 0s and 1s allowed.
        if len(sample_data) % bits_per_symbol != 0 or any(ch not in ['0', '1'] for ch in sample_data):
            calc_values = {"error": f"Sample data length must be a multiple of {bits_per_symbol} bits, and only 0 or 1 are allowed."}
        else:
            # Basic parameters.
            bandwidth_required = target_data_rate / bits_per_symbol
            snr_req = bits_per_symbol + 5
            baud_rate = target_data_rate / bits_per_symbol

            # Compute effective throughput.
            if snr < snr_req:
                effective_throughput = target_data_rate * (snr / snr_req)
            else:
                effective_throughput = target_data_rate

            calc_values = {
                "SNR Requirement": f"{snr_req:.2f} dB",
                "Bandwidth Required": f"{bandwidth_required:.2f} MHz",
                "Bits per Symbol": bits_per_symbol,
                "Baud Rate": f"{baud_rate:.2f} symbols/s",
                "Effective Throughput": f"{effective_throughput:.2f} Mbps"
            }

            if mod_type in STANDARD_VALUES and mod_order in STANDARD_VALUES[mod_type]:
                std = STANDARD_VALUES[mod_type][mod_order]
                standard_values = {
                    "SNR Req (Std)": f"{std['snr_req']} dB",
                    "BW (Std)": f"{std['bw_req']} MHz",
                    "Bits/Symbol (Std)": std['bits_per_symbol'],
                    "Baud Rate (Std)": f"{std['baud_rate']} symbols/s",
                    "Std Throughput": f"{std['throughput']} bps"
                }
            else:
                standard_values = {
                    "SNR Req (Std)": "N/A",
                    "BW (Std)": "N/A",
                    "Bits/Symbol (Std)": "N/A",
                    "Baud Rate (Std)": "N/A",
                    "Std Throughput": "N/A"
                }

            samples_per_symbol = 100
            num_symbols = len(sample_data) // bits_per_symbol
            T_symbol = 1.0 / baud_rate if baud_rate != 0 else 1.0

            # Helper: generate a symbol's waveform based on modulation type.
            def generate_symbol_wave(mod_type, symbol_bits, samples_per_symbol, T_symbol):
                t = [i / samples_per_symbol * T_symbol for i in range(samples_per_symbol)]
                if mod_type == "QAM":
                    if len(symbol_bits) >= 2:
                        half = len(symbol_bits) // 2
                        I_bits = symbol_bits[:half]
                        Q_bits = symbol_bits[half:]
                        I_val = int(I_bits, 2)
                        Q_val = int(Q_bits, 2) if Q_bits else 0
                        I_max = (2**len(I_bits) - 1) if len(I_bits) > 0 else 1
                        Q_max = (2**len(Q_bits) - 1) if len(Q_bits) > 0 else 1
                        I_amp = 2 * I_val / I_max - 1 if I_max != 0 else 0
                        Q_amp = 2 * Q_val / Q_max - 1 if Q_max != 0 else 0
                        f = 1000
                        return [I_amp * math.cos(2 * math.pi * f * x) - Q_amp * math.sin(2 * math.pi * f * x) for x in t]
                    else:
                        A = 1 if symbol_bits == "1" else -1
                        f = 1000
                        return [A * math.cos(2 * math.pi * f * x) for x in t]
                elif mod_type == "PSK":
                    phase = int(symbol_bits, 2) / (2**len(symbol_bits)) * 2 * math.pi
                    f = 1000
                    return [math.cos(2 * math.pi * f * x + phase) for x in t]
                elif mod_type == "FSK":
                    freq_offset = int(symbol_bits, 2)
                    base_freq = 1000
                    f_symbol = base_freq + freq_offset * 50
                    return [math.cos(2 * math.pi * f_symbol * x) for x in t]
                elif mod_type == "ASK":
                    A = 1 if symbol_bits == "1" else 0.5
                    f = 1000
                    return [A * math.cos(2 * math.pi * f * x) for x in t]
                elif mod_type == "CSS":
                    f0 = 800
                    f1 = 1200
                    phase_shift = int(symbol_bits, 2) / (2**len(symbol_bits)) * math.pi
                    return [math.cos(2 * math.pi * (f0 * x + (f1 - f0) / (2 * T_symbol) * x * x) + phase_shift) for x in t]
                else:
                    f = 1000
                    return [math.cos(2 * math.pi * f * x) for x in t]

            # Process each symbol from the input bit stream.
            for s in range(num_symbols):
                symbol_bits = sample_data[s * bits_per_symbol:(s + 1) * bits_per_symbol]
                symbol_wave = generate_symbol_wave(mod_type, symbol_bits, samples_per_symbol, T_symbol)
                for i in range(samples_per_symbol):
                    t_val = s * T_symbol + (i / samples_per_symbol) * T_symbol
                    time_values.append(t_val)
                    time_domain_data.append(symbol_wave[i])
            index_labels = list(range(len(time_domain_data)))

            # Frequency spectrum (placeholder)
            for i in range(20):
                freq_data.append({"freq": i, "amp": random.uniform(0, 1)})

            # BER vs SNR (placeholder)
            for i in range(5, 31, 5):
                ber_val = 1.0 / (10**(i / 10))
                ber_snr_data.append({"snr": i, "ber": ber_val})

            # Constellation (placeholder)
            for _ in range(mod_order):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                constellation_data.append({"x": x, "y": y})

            # Acceptable Noise Level with random tolerance.
            for angle_deg in range(0, 360, 30):
                angle = math.radians(angle_deg)
                random_offset = random.uniform(-0.2, 0.2)
                radius = 1 + random_offset
                noise_data.append({"x": radius * math.cos(angle), "y": radius * math.sin(angle)})

    return render_template(
        "index.html",
        mod_type=mod_type,
        mod_order=mod_order,
        snr=snr,
        target_data_rate=target_data_rate,
        sample_data=sample_data,
        auto_compute=auto_compute,
        calc_values=calc_values,
        standard_values=standard_values,
        time_domain_data=time_domain_data,
        time_values=time_values,
        index_labels=index_labels,
        freq_data=freq_data,
        ber_snr_data=ber_snr_data,
        constellation_data=constellation_data,
        noise_data=noise_data
    )

if __name__ == "__main__":
    app.run(debug=True)
