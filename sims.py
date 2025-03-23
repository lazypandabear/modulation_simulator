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

        # Validate sample data
        if len(sample_data) != bits_per_symbol or any(ch not in ['0','1'] for ch in sample_data):
            calc_values = {"error": f"Sample data must be {bits_per_symbol} bits of 0 or 1."}
        else:
            bandwidth_required = target_data_rate / bits_per_symbol
            snr_req = bits_per_symbol + 5
            baud_rate = target_data_rate / bits_per_symbol
            computed_throughput = target_data_rate

            calc_values = {
                "SNR Requirement": f"{snr_req:.2f} dB",
                "Bandwidth Required": f"{bandwidth_required:.2f} MHz",
                "Bits per Symbol": bits_per_symbol,
                "Baud Rate": f"{baud_rate:.2f} symbols/s",
                "Calculated Throughput": f"{computed_throughput:.2f} Mbps"
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

            # Time domain data (simple sine wave plus bit offset)
            for i, bit in enumerate(sample_data):
                val = math.sin(2 * math.pi * i / bits_per_symbol) + int(bit)
                time_domain_data.append(val)

            # Frequency spectrum data (placeholder)
            for i in range(20):
                freq_data.append({
                    "freq": i,
                    "amp": random.uniform(0, 1)
                })

            # BER vs. SNR data (placeholder)
            for i in range(5, 31, 5):
                # Typical shape: BER ~ 1 / (SNR^some_factor)
                ber_val = 1.0 / (10**(i/10))
                ber_snr_data.append({"snr": i, "ber": ber_val})

            # Constellation data (placeholder for QAM)
            for _ in range(mod_order):
                # Randomly scatter points
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                constellation_data.append({"x": x, "y": y})

            # Acceptable noise level data (placeholder)
            # We'll create small circle around origin
            for angle_deg in range(0, 360, 30):
                angle = math.radians(angle_deg)
                radius = 1
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                noise_data.append({"x": x, "y": y})

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
        freq_data=freq_data,
        ber_snr_data=ber_snr_data,
        constellation_data=constellation_data,
        noise_data=noise_data
    )

if __name__ == "__main__":
    app.run(debug=True)
