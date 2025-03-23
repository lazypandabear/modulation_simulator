# Modulator Simulator

This project is a digital modulation simulation application built using Python and Flask. It simulates various digital modulation schemes (QAM, PSK, FSK, ASK, CSS) and displays key performance metrics and plots such as:

- **Time Domain Signal**
- **Frequency Spectrum** (using Welch’s method)
- **BER vs. SNR**
- **SNR vs. Throughput**
- **Constellation Map** (with highlighted input symbols)
- **Noise Tolerance Map**

## Requirements

- Python 3.x
- [Flask](https://flask.palletsprojects.com/)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- Chart.js (included via CDN in the HTML template)

Install the Python dependencies using pip:

```bash
pip install flask numpy scipy


Project Files
sims.py:
The main Flask application that processes user input, computes simulation results, and serves the results and charts to the web page.

templates/index.html:
The HTML template that includes the input form and displays the charts (using Chart.js) for the simulation data.

Running the Application
Open a terminal and navigate to the project directory.

Run the Flask application with:

bash
Copy
python sims.py
Open your web browser and go to http://localhost:5000.

How to Use
Input Section
Type of Digital Modulation:
Choose between QAM, PSK, FSK, ASK, or CSS using the radio buttons. The label and buttons are aligned to the left.

Modulation Order (2^x):
Select the modulation order (e.g., 2, 4, 8, etc.). The number of bits per symbol is determined by the logarithm of the modulation order.

SNR (dB):
Enter the signal-to-noise ratio in dB.

Target Data Rate (Mbps):
Input the desired throughput in Mbps.

Sample Data Stream (bits):
Provide a bit stream. The length must be a multiple of the number of bits per symbol (determined by the modulation order).

Auto Compute:
Check this option to enable automatic computation when inputs change.

Click Run Simulation to start the simulation.

Result Section
This section displays:

Calculated simulation parameters such as SNR requirement, bandwidth, bits per symbol, baud rate, and effective throughput.

Standard values for the selected modulation scheme.

Chart Section
The charts are divided as follows:

2×2 Grid:

Time Domain Signal: Plot of the generated time-domain waveform.

Frequency Spectrum: The power spectral density (in dB/Hz) estimated using Welch’s method.

BER vs. SNR: A placeholder plot showing the relationship between SNR and Bit Error Rate.

SNR vs. Throughput: A line chart showing how throughput varies with SNR.

Full-width Charts:

Constellation Map: Displays all possible constellation points along with the actual input symbols (highlighted in blue and larger in size). Cross lines indicate the in-phase and quadrature axes.

Noise Tolerance Map: Displays circles (tolerance boundaries) around each constellation point to indicate possible noise tolerance.

Customization
Modify the simulation formulas or parameters in sims.py as needed.

Adjust the Chart.js configuration in templates/index.html to customize chart appearance, axis limits, or other settings.

Further refine the noise tolerance and constellation mapping logic to suit your simulation requirements.

Troubleshooting
Charts Not Displaying:
Open your browser’s developer console to check for any errors. Ensure that the backend is providing non-empty data arrays for each chart.

Input Issues:
Verify that the sample data stream meets the requirement (length must be a multiple of bits per symbol).

FFT/PSD Problems:
Ensure that enough time-domain samples are generated for the FFT used in Welch’s method.

License

This project is provided "as is" without any warranty.