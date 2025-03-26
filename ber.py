import math
from scipy.special import erfc

def compute_ber_bpsk(snr_db):
    """
    Compute BER for BPSK over an AWGN channel.
    
    Formula: 
      BER = 0.5 * erfc( sqrt(SNR_linear) )
    """
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * erfc(math.sqrt(snr_lin))

def compute_ber_qam(snr_db, M):
    """
    Compute approximate BER for square M-QAM with Gray coding over an AWGN channel.
    
    Formula (approximate):
      BER ≈ [4(1 - 1/√M)/log2(M)] * Q( sqrt(3 * SNR_linear/(M-1)) )
    where Q(x) = 0.5 * erfc(x/√2) and SNR_linear = 10^(snr_db/10)
    """
    snr_lin = 10 ** (snr_db / 10)
    k = math.log2(M)
    Q_arg = math.sqrt(3 * snr_lin / (M - 1))
    ber = (4 * (1 - 1 / math.sqrt(M)) / k) * 0.5 * erfc(Q_arg / math.sqrt(2))
    return ber

def compute_ber_ask(snr_db):
    """
    Compute BER for binary ASK (coherent detection) over AWGN.
    
    Formula:
      BER ≈ Q( sqrt(SNR_linear) ) = 0.5 * erfc( sqrt(SNR_linear)/√2 )
    """
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * erfc(math.sqrt(snr_lin) / math.sqrt(2))

def compute_ber_fsk(snr_db):
    """
    Compute BER for binary non-coherent FSK over AWGN.
    
    Formula (approximate):
      BER ≈ 0.5 * exp(-SNR_linear/2)
    """
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * math.exp(-snr_lin / 2)

def compute_ber_fsk_m(snr_db, M):
    """
    Compute approximate BER for non-coherent M-ary FSK over AWGN.
    
    Formula (approximate):
      BER ≈ ((M-1)/(2 * log2(M))) * exp(-SNR_linear/2)
    """
    snr_lin = 10 ** (snr_db / 10)
    k = math.log2(M)
    return ((M - 1) / (2 * k)) * math.exp(-snr_lin / 2)

def compute_ber_css(snr_db, SF):
    """
    Compute an approximate BER for LoRa's Chirp Spread Spectrum (CSS) modulation.
    
    In LoRa, the spreading factor (SF) defines the number of orthogonal chirp signals:
      M = 2^SF
    CSS detection is non-coherent, and a common approximation for non-coherent orthogonal modulation
    (similar to M-ary FSK) is:
    
      BER ≈ ((M-1) / (2 * SF)) * exp(-SNR_linear/2)
    
    where SNR_linear = 10^(snr_db/10).
    
    Note: This formula is an approximation and may not capture all aspects of CSS performance.
    """
    M = 2 ** SF
    snr_lin = 10 ** (snr_db / 10)
    ber = ((M - 1) / (2 * SF)) * math.exp(-snr_lin / 2)
    return ber

# Example usage:
if __name__ == "__main__":
    SNR_dB_values = [5, 10, 15, 20, 25, 30]
    
    print("BPSK BER:")
    for snr_db in SNR_dB_values:
        print(f"SNR {snr_db} dB: {compute_ber_bpsk(snr_db):.3e}")
    
    mod_order_qam = 1024
    print("\n1024-QAM BER:")
    for snr_db in SNR_dB_values:
        print(f"SNR {snr_db} dB: {compute_ber_qam(snr_db, mod_order_qam):.3e}")
    
    print("\nBinary ASK BER:")
    for snr_db in SNR_dB_values:
        print(f"SNR {snr_db} dB: {compute_ber_ask(snr_db):.3e}")
    
    print("\nBinary FSK BER:")
    for snr_db in SNR_dB_values:
        print(f"SNR {snr_db} dB: {compute_ber_fsk(snr_db):.3e}")
    
    mod_order_fsk = 4  # Example for 4-FSK
    print("\n4-FSK BER:")
    for snr_db in SNR_dB_values:
        print(f"SNR {snr_db} dB: {compute_ber_fsk_m(snr_db, mod_order_fsk):.3e}")
    
    SF = 7  # Typical spreading factor for LoRa (7 to 12)
    print("\nLoRa CSS BER (SF=7):")
    for snr_db in SNR_dB_values:
        print(f"SNR {snr_db} dB: {compute_ber_css(snr_db, SF):.3e}")
