import math

def get_symbol_coord(mod_type, symbol_bits):
    """
    Computes the (x, y) coordinate for a given symbol's bit string based on the modulation type.
    
    Parameters:
        mod_type (str): The modulation type. Supported types: 'ASK', 'FSK', 'PSK', 'QAM', 'CSS'.
        symbol_bits (str): A string of bits representing the symbol (e.g., "1010").
    
    Returns:
        tuple: (x, y) coordinate corresponding to the symbol.
    
    Mapping Details:
      - ASK: Maps the integer value of the bits linearly to an amplitude in [-1, 1] (real axis).
      - FSK: Maps the integer value to a phase angle on a circle.
      - PSK: Maps the bit string to a phase angle (for BPSK, QPSK, etc.).
      - QAM: Splits the bits into two halves for the I and Q components.
      - CSS (Chirp Spread Spectrum as used in LoRaWAN): 
             Typically uses a spreading factor (SF) where M = 2^(SF). Here, we map the symbol 
             to a phase on a circle.
    """
    mod_type = mod_type.upper()
    bits_len = len(symbol_bits)
    
    if mod_type == "QAM":
        # For QAM, if more than one bit is available, split the bits equally for I and Q.
        if bits_len >= 2:
            half = bits_len // 2 # Split the bits into two halves
            I_bits = symbol_bits[:half] # First half for I
            Q_bits = symbol_bits[half:] # Second half for Q
            I_val = int(I_bits, 2)  # Convert bits to integer
            Q_val = int(Q_bits, 2) if Q_bits else 0 # Convert bits to integer
            I_max = (2**len(I_bits) - 1) if len(I_bits) > 0 else 1 # Max value for normalization
            Q_max = (2**len(Q_bits) - 1) if len(Q_bits) > 0 else 1 # Max value for normalization
            # Map to [-1, 1]
            I_amp = 2 * I_val / I_max - 1 if I_max != 0 else 0  # Normalize to [-1, 1]
            Q_amp = 2 * Q_val / Q_max - 1 if Q_max != 0 else 0   # Normalize to [-1, 1]
            # Return (I, Q) with Q inverted to match common convention
            return (I_amp, -Q_amp)
        else:
            # If only one bit, simply use +/-1.
            return (1, 0) if symbol_bits == "1" else (-1, 0)
    
    elif mod_type == "PSK":
        # For PSK, convert the bit string to an integer and map it to a phase.
        phase = int(symbol_bits, 2) / (2 ** bits_len) * 2 * math.pi
        return (math.cos(phase), math.sin(phase))
    
    elif mod_type == "FSK":
        # For FSK, treat the bit string as an integer index and map it on a circle.
        #FSK doesn't inherently encode information in phase (or angle) like PSK. 
        # In FSK, data is represented by different frequencies. #
        # However, for visualization purposes, it's common to map FSK symbols onto a circle, 
        # assigning each frequency an angle so they appear as distinct points. 
        # This mapping is a convenient representation rather than an intrinsic part of FSK modulation.
        # Assume M = 2^(bits_len) for simplicity.
        M = 2 ** bits_len
        index = int(symbol_bits, 2)
        angle = 2 * math.pi * index / M
        return (math.cos(angle), math.sin(angle))
    
    elif mod_type == "ASK":
        # For ASK, map the integer value linearly to an amplitude on the real axis.
        val = int(symbol_bits, 2)
        max_val = (2 ** bits_len - 1) if bits_len > 0 else 1
        amplitude = 2 * val / max_val - 1  # Maps to [-1, 1]
        return (amplitude, 0)
    
    elif mod_type == "CSS":
        # For CSS (LoRaWAN style), the spreading factor SF is typically the number of bits.
        # With M = 2^(SF), map the symbol to a phase on a unit circle.
        M = 2 ** bits_len
        phase = 2 * math.pi * int(symbol_bits, 2) / M
        return (math.cos(phase), math.sin(phase))
    
    else:
        # Unknown modulation type.
        return (0, 0)

if __name__ == "__main__":
    # Example usage: test get_symbol_coord for each modulation type with a sample bit string.
    modulations = ['ASK', 'FSK', 'PSK', 'QAM', 'CSS']
    sample_symbol = "1010"  # 4-bit example symbol
    for mod in modulations:
        coord = get_symbol_coord(mod, sample_symbol)
        print(f"{mod}: Symbol {sample_symbol} maps to coordinate {coord}")
