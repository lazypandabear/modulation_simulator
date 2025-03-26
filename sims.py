import math
import random
import numpy as np
from scipy.signal import welch
from flask import Flask, render_template, request
from scipy.special import erfcinv
from ber import compute_ber_bpsk, compute_ber_qam, compute_ber_ask, compute_ber_fsk, compute_ber_fsk_m, compute_ber_css
from modulation import get_symbol_coord

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

import math
from scipy.special import erfcinv

def compute_required_snr(mod_type, mod_order, bits_per_symbol, target_ser=1e-3):
    """
    Compute the required SNR (in dB) for a given modulation type and modulation order,
    based on an approximate target symbol error rate (SER).
    
    Supported mod_type: 'QAM', 'PSK', 'FSK'.
    For QAM and PSK, we use Q-function approximations.
    For FSK, we use a non-coherent M-ary FSK approximation.
    
    Parameters:
        mod_type (str): 'QAM', 'PSK', 'FSK' (case-insensitive).
        mod_order (int): e.g., 16, 64, 1024, etc.
        target_ser (float): desired symbol error rate. default=1e-3
    
    Returns:
        float: Required SNR in dB.
    
    Raises:
        ValueError: If modulation type is not recognized or the formula fails.
    """
    mod_type = mod_type.upper()
    
    # Qinv helper for QAM/PSK
    def Qinv(x):
        # Q^-1(x) = sqrt(2)*erfcinv(2x)
        from math import sqrt
        from scipy.special import erfcinv
        return math.sqrt(2) * erfcinv(2*x)
    
    if mod_type == "QAM":
        # M-ary QAM formula:
        # SNR_req_linear ≈ (M-1)/3 * [Q^-1( target_ser / (4(1-1/√M)) )]^2
        M = mod_order
        if M < 2:
            raise ValueError("QAM order must be >= 2")
        numerator = Qinv(target_ser / (4 * (1 - 1/math.sqrt(M))))
        snr_linear = (M - 1)/3 * (numerator**2)
        snr_db = 10*math.log10(snr_linear)
        return snr_db
    
    elif mod_type == "PSK":
        # M-ary PSK formula (approx):
        # SNR_req_linear ≈ [ Q^-1( target_ser/2 ) / ( sqrt(2)*sin(pi/M) ) ]^2
        M = mod_order
        if M < 2:
            raise ValueError("PSK order must be >= 2")
        # Special case: if M=2 => BPSK
        if M == 2:
            # For BPSK, SER = BER, SER= Q(sqrt(2*SNR)), solve for SNR
            # approximate symbol error = bit error
            # SER = target_ser => Q^-1(target_ser) = sqrt(2*SNR)
            # SNR = [Q^-1(target_ser)]^2 / 2
            qv = Qinv(target_ser)
            snr_linear = (qv**2)/2
            return 10*math.log10(snr_linear)
        else:
            # M>2
            qv = Qinv(target_ser/2)
            denom = math.sqrt(2)*math.sin(math.pi/M)
            snr_linear = (qv/denom)**2
            return 10*math.log10(snr_linear)
    
    elif mod_type == "FSK":
        # Non-coherent M-ary FSK approximation:
        # SER ≈ (M-1)/2 * exp(-SNR/2).
        # Solve for SNR given target_ser => exp(-SNR/2) = SER * 2/(M-1)
        # => SNR = -2 ln(SER * 2/(M-1))
        M = mod_order
        if M < 2:
            raise ValueError("FSK order must be >= 2")
        # Solve for SNR in linear scale
        # target_ser = (M-1)/2 * exp(-SNR/2)
        # => exp(-SNR/2) = target_ser*2/(M-1)
        argument = target_ser * 2.0/(M - 1)
        if argument <= 0:
            raise ValueError("Invalid target_ser or M-1 for FSK formula.")
        # SNR must be positive => if argument >= 1 => negative or zero => no real solution
        if argument >= 1:
            # Means you'd need infinite SNR or we can't solve
            # Return something large or handle gracefully
            return float('inf')
        # Otherwise:
        snr_linear = -2 * math.log(argument)
        # Convert to dB
        snr_db = 10 * math.log10(snr_linear)
        return snr_db
    
    elif mod_type == "ASK":
        # Coherent M-ary ASK approximation:
        # SER ≈ [(2*(M-1))/(M*log2(M))] * Q(sqrt(3*SNR_linear/(M^2-1)))
        M = mod_order
        if M < 2:
            raise ValueError("ASK order must be >= 2")
        
        # Special case: if M=2 => Binary ASK
        if M == 2:
            # For Binary ASK, SER = Q(sqrt(SNR_linear))
            # approximate symbol error = bit error
            # SER = target_ser => Q^-1(target_ser) = sqrt(SNR_linear)
            # SNR_linear = [Q^-1(target_ser)]^2
            qv = Qinv(target_ser)
            snr_linear = qv**2
            return 10*math.log10(snr_linear)
        else:
            # M>2
            # SER ≈ [(2*(M-1))/(M*log2(M))] * Q(sqrt(3*SNR_linear/(M^2-1)))
            # Q(x) = 0.5 * erfc(x/sqrt(2))
            # Qinv(x) = sqrt(2) * erfcinv(2*x)
            # target_ser = [(2*(M-1))/(M*log2(M))] * Q(sqrt(3*SNR_linear/(M^2-1)))
            # target_ser / [(2*(M-1))/(M*log2(M))] = Q(sqrt(3*SNR_linear/(M^2-1)))
            # Qinv(target_ser / [(2*(M-1))/(M*log2(M))]) = sqrt(3*SNR_linear/(M^2-1))
            # Qinv(target_ser / [(2*(M-1))/(M*log2(M))])^2 = 3*SNR_linear/(M^2-1)
            # SNR_linear = (M^2-1) * Qinv(target_ser / [(2*(M-1))/(M*log2(M))])^2 / 3
            
            qv = Qinv(target_ser / ((2*(M-1))/(M*bits_per_symbol)))
            snr_linear = ((M**2)-1) * (qv**2) / 3
            return 10*math.log10(snr_linear)

    elif mod_type == "CSS":
        # Non-coherent CSS (LoRa-like) approximation:
        # BER ≈ ((M-1) / (2 * SF)) * exp(-SNR_linear/2)
        # where M = 2^SF
        SF = bits_per_symbol  # Spreading factor is the number of bits per symbol
        M = mod_order
        if M < 2:
            raise ValueError("CSS order must be >= 2")
        
        # Solve for SNR in linear scale
        # target_ser = ((M-1) / (2 * SF)) * exp(-SNR/2)
        # => exp(-SNR/2) = target_ser * (2 * SF) / (M-1)
        argument = target_ser * (2 * SF) / (M - 1)
        if argument <= 0:
            raise ValueError("Invalid target_ser or M-1 for CSS formula.")
        # SNR must be positive => if argument >= 1 => negative or zero => no real solution
        if argument >= 1:
            # Means you'd need infinite SNR or we can't solve
            # Return something large or handle gracefully
            return float('inf')
        # Otherwise:
        snr_linear = -2 * math.log(argument)
        # Convert to dB
        snr_db = 10 * math.log10(snr_linear)
        return snr_db
    
    else:
        raise ValueError("Modulation type not supported.")


def get_constellation_points(mod_type, mod_order):
    """Deterministic set of constellation points (no randomness)."""
    points = []
    bits_per_symbol = int(math.log2(mod_order))

    if mod_type == "QAM":
        # For square QAM: place points on a grid if perfect square, else on a circle
        size = int(math.sqrt(mod_order))
        if size * size == mod_order:
            step = 2 / (size - 1) if size > 1 else 1
            start = -1
            for i in range(size):
                for j in range(size):
                    x = start + i * step
                    y = start + j * step
                    points.append({"x": x, "y": y})
        else:
            # If not perfect square, place them around a circle
            for i in range(mod_order):
                angle = 2 * math.pi * i / mod_order
                x = math.cos(angle)
                y = math.sin(angle)
                points.append({"x": x, "y": y})

    elif mod_type == "PSK":
        # Place mod_order points equally on a unit circle
        for i in range(mod_order):
            angle = 2 * math.pi * i / mod_order
            x = math.cos(angle)
            y = math.sin(angle)
            points.append({"x": x, "y": y})

    elif mod_type == "FSK":
        # Also place points on a circle
        for i in range(mod_order):
            angle = 2 * math.pi * i / mod_order
            x = math.cos(angle)
            y = math.sin(angle)
            points.append({"x": x, "y": y})

    elif mod_type == "ASK":
        # For ASK, place points on the real axis from -1 to +1
        step = 2 / (mod_order - 1) if mod_order > 1 else 1
        start = -1
        for i in range(mod_order):
            x = start + i * step
            points.append({"x": x, "y": 0})

    elif mod_type == "CSS":
        # Place them on a small spiral
        for i in range(mod_order):
            angle = 2 * math.pi * i / mod_order
            radius = 1 + 0.2 * i
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append({"x": x, "y": y})

    return points

def generate_symbol_wave(mtype, symbol_bits, sps, T):
    """
    Generate a single symbol's time-domain waveform for the given modulation.
    """
    import math
    t = [i / sps * T for i in range(sps)]

    if mtype == "QAM":
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
            return [
                I_amp * math.cos(2 * math.pi * f * x)
                - Q_amp * math.sin(2 * math.pi * f * x)
                for x in t
            ]
        else:
            A = 1 if symbol_bits == "1" else -1
            f = 1000
            return [A * math.cos(2 * math.pi * f * x) for x in t]

    elif mtype == "PSK":
        phase = int(symbol_bits, 2) / (2**len(symbol_bits)) * 2 * math.pi
        f = 1000
        return [math.cos(2 * math.pi * f * x + phase) for x in t]

    elif mtype == "FSK":
        freq_offset = int(symbol_bits, 2)
        base_freq = 1000
        f_symbol = base_freq + freq_offset * 50
        return [math.cos(2 * math.pi * f_symbol * x) for x in t]

    elif mtype == "ASK":
        A = 1 if symbol_bits == "1" else 0.5
        f = 1000
        return [A * math.cos(2 * math.pi * f * x) for x in t]

    elif mtype == "CSS":
        f0 = 800
        f1 = 1200
        phase_shift = int(symbol_bits, 2) / (2**len(symbol_bits)) * math.pi
        return [
            math.cos(
                2 * math.pi * (f0 * x + (f1 - f0) / (2 * T) * x * x)
                + phase_shift
            )
            for x in t
        ]

    else:
        f = 1000
        return [math.cos(2 * math.pi * f * x) for x in t]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    mod_type = "QAM"
    mod_order = 2
    snr = 10
    target_data_rate = 100.0 # Mbps
    sample_data = "0"
    auto_compute = False

    calc_values = {}
    standard_values = {}
    time_domain_data = []
    time_values = []
    freq_data = []
    ber_snr_data = []
    constellation_data = []
    noise_data = []
    snr_throughput_data = []

    if request.method == 'POST':
        mod_type = request.form.get("mod_type", "QAM")
        mod_order = int(request.form.get("mod_order", 2))
        snr = float(request.form.get("snr", 10))
        target_data_rate = float(request.form.get("target_data_rate", 1.0))
        sample_data = request.form.get("sample_data", "")
        auto_compute = request.form.get("auto_compute") == "on"

        bits_per_symbol = int(math.log2(mod_order))

        # Validate input bits
        if len(sample_data) % bits_per_symbol != 0 or any(ch not in ['0','1'] for ch in sample_data):
            calc_values = {
                "error": f"Sample data length must be multiple of {bits_per_symbol}, only 0/1 allowed."
            }
        else:
            # Basic parameters
            bandwidth_required = target_data_rate / bits_per_symbol
            try:
                # Attempt to compute required SNR
                snr_req = compute_required_snr(mod_type, mod_order, bits_per_symbol, target_ser=1e-3)
            except ValueError as e:
                # If the function raises ValueError, store the message
                calc_values["error"] = f"Failed to compute SNR: {str(e)}" 
            
            #simple computation is snr_req  = bits_per_symbol + 5 The extra 5 dB is an empirical margin that accounts for practical system losses, 
            #non-idealities, and other implementation factors. It provides a rough baseline to ensure that the SNR is sufficiently above the minimum required level for reliable detection.
            
            
            baud_rate = target_data_rate / bits_per_symbol

            # Effective throughput
            if snr < snr_req:
                effective_throughput = target_data_rate * (snr / snr_req)
            else:
                effective_throughput = target_data_rate

            calc_values = {
                "SNR Requirement": f"{snr_req:.2f} dB",
                "Bandwidth Required": f"{bandwidth_required:.2f} MHz",
                "Bits per Symbol": bits_per_symbol,
                "Baud Rate": f"{baud_rate:.2f} Millions symbols/s",
                "Effective Throughput": f"{effective_throughput:.2f} Mbps"
            }

            # Standard values if available
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

            # Generate time-domain data
            samples_per_symbol = 100
            num_symbols = len(sample_data) // bits_per_symbol
            T_symbol = 1.0 / baud_rate if baud_rate != 0 else 1.0

            # Build entire time-domain signal
            for s in range(num_symbols):
                sym_bits = sample_data[s*bits_per_symbol : (s+1)*bits_per_symbol]
                wave = generate_symbol_wave(mod_type, sym_bits, samples_per_symbol, T_symbol)
                for i, val in enumerate(wave):
                    t_val = s * T_symbol + (i / samples_per_symbol) * T_symbol
                    time_values.append(t_val)
                    time_domain_data.append(val)

            # 1) Frequency Spectrum using Welch's method (similar to pwelch in MATLAB)
            #    Make sure we have at least 2 samples to do an FFT
            if len(time_domain_data) >= 2:
                # Approximate sample rate
                # If time_values are uniform, we can do:
                dt = (time_values[-1] - time_values[0]) / (len(time_values) - 1) if len(time_values) > 1 else 1
                fs = 1 / dt if dt != 0 else 1

                f, Pxx = welch(
                    x=time_domain_data,
                    fs=fs,
                    window='hann',
                    nperseg=min(1024, len(time_domain_data)),  # or pick your segment size
                    noverlap=None,
                    scaling='density',
                    return_onesided=False  # to get a "centered" spectrum
                )
                # Shift so that 0 Hz is in the middle, as "centered" would in MATLAB
                fshift = np.fft.fftshift(f)
                Pxx_shift = np.fft.fftshift(Pxx)
                # Convert to dB/Hz
                Pxx_dB = 10 * np.log10(Pxx_shift, out=np.zeros_like(Pxx_shift), where=(Pxx_shift>0))

                # Prepare freq_data for the frontend
                freq_data = []
                for freq_val, pwr_val in zip(fshift, Pxx_dB):
                    freq_data.append({"freq": float(freq_val), "amp": float(pwr_val)})
            else:
                # Not enough samples for FFT, provide empty or minimal data
                freq_data = []

            # 2) BER vs SNR (placeholder)
            '''
            this is for simple approach
            ber_snr_data = []
            for i in range(5, 31, 5):
                ber_val = 1.0 / (10**(i / 10))
                ber_snr_data.append({"snr": i, "ber": ber_val})'''
            
            # Compute BER for the given modulation type and order
            ber_snr_data = []
            for snr_db in range(5, 31, 5):
                if mod_type.upper() == "QAM":
                    ber_val = compute_ber_qam(snr_db, mod_order)
                elif mod_type.upper() == "PSK":
                    if mod_order == 2:
                        ber_val = compute_ber_bpsk(snr_db)
                    else:
                        # Extend for higher-order PSK if needed
                        ber_val = compute_ber_bpsk(snr_db)  # As a simple approximation
                elif mod_type.upper() == "ASK":
                    ber_val = compute_ber_ask(snr_db)
                elif mod_type.upper() == "FSK":
                    if mod_order == 2:
                        ber_val = compute_ber_fsk(snr_db)
                    else:
                        ber_val = compute_ber_fsk_m(snr_db, mod_order)
                elif mod_type.upper() == "CSS":
                    # Use the spreading factor for CSS; for instance, SF=7.
                    SF = 7  # You may allow this to be an input parameter
                    ber_val = compute_ber_css(snr_db, SF)
                else:
                    ber_val = 1.0 / (10 ** (snr_db / 10))  # fallback placeholder
                
                ber_snr_data.append({"snr": snr_db, "ber": float(ber_val)})


            # 3) Deterministic constellation
            constellation_data = get_constellation_points(mod_type, mod_order)

            # 4) Noise boundary:
            # Assume:
            # - sample_data is a string containing the transmitted bit stream.
            # - bits_per_symbol is computed as: bits_per_symbol = int(math.log2(mod_order))
            # - get_symbol_coord(mod_type, symbol_bits) returns a tuple (x, y) for that symbol.

            # Parse sample_data into symbols
            parsed_symbols = [sample_data[i:i+bits_per_symbol] for i in range(0, len(sample_data), bits_per_symbol)]

            radius = 0.2
            noise_data = []

            for idx, symbol_bits in enumerate(parsed_symbols):
                # Compute the (x, y) coordinate for the actual symbol using get_symbol_coord
                cpoint = get_symbol_coord(mod_type, symbol_bits)
                circle_points = []
                for angle_deg in range(0, 361, 10):
                    angle = math.radians(angle_deg)
                    x = cpoint[0] + radius * math.cos(angle)
                    y = cpoint[1] + radius * math.sin(angle)
                    circle_points.append({"x": x, "y": y})
                noise_data.append({
                    "label": f"Tolerance Circle {idx}",
                    "center": {"x": cpoint[0], "y": cpoint[1]},
                    "points": circle_points
                })

            # 5) SNR vs. Throughput line chart
            snr_throughput_data = []
            for test_snr in range(0, 35, 5):
                if test_snr < snr_req:
                    thr = target_data_rate * (test_snr / snr_req)
                else:
                    thr = target_data_rate
                snr_throughput_data.append({"snr": test_snr, "throughput": thr})

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
        freq_data=freq_data,
        ber_snr_data=ber_snr_data,
        constellation_data=constellation_data,
        noise_data=noise_data,
        snr_throughput_data=snr_throughput_data
    )

if __name__ == "__main__":
    app.run(debug=True)
