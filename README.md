# Modulator Simulator

This project is a digital modulation simulation application built using Python and Flask. It simulates various digital modulation schemes (QAM, PSK, FSK, ASK, CSS) and displays key performance metrics and plots such as:

- **Time Domain Signal:**  Visualizes the waveform of the modulated signal over time.
- **Frequency Spectrum:** Shows the power spectral density (PSD) of the signal, calculated using Welch's method.
- **BER vs. SNR:**  Illustrates the relationship between Bit Error Rate (BER) and Signal-to-Noise Ratio (SNR) for the selected modulation scheme.
- **SNR vs. Throughput:**  Demonstrates how the effective data throughput changes with varying SNR levels.
- **Constellation Map:**  Displays the constellation diagram, showing the possible symbol locations in the complex plane. Input symbols are highlighted.
- **Noise Tolerance Map:**  Visualizes the noise tolerance boundaries around each constellation point, indicating the region where noise can be added without causing symbol errors.
- **Calculated Values:** Displays key metrics like SNR requirement, bandwidth, bits per symbol, baud rate, and effective throughput.
- **Standard Values:** Compares the calculated values with standard values for the selected modulation scheme and order.

## Features

- **Multiple Modulation Types:** Supports QAM, PSK, FSK, ASK, and CSS modulation schemes.
- **Configurable Modulation Order:** Allows selection of various modulation orders (e.g., 2, 4, 8, 16, ..., 65536).
- **SNR and Data Rate Control:** Users can specify the desired SNR and target data rate.
- **Custom Bit Stream Input:**  Users can input a custom bit stream to simulate.
- **Auto-Compute Option:** Automatically recalculates and updates the results when input parameters change.
- **Interactive Charts:** Utilizes Chart.js to provide dynamic and informative visualizations.
- **BER Calculation:** Computes the Bit Error Rate (BER) based on the selected modulation type and SNR.
- **Constellation and Noise Visualization:** Provides clear constellation diagrams and noise tolerance boundaries.
- **Frequency Spectrum Analysis:** Uses Welch's method for accurate power spectral density estimation.
- **Standard Value Comparison:** Shows standard values for comparison with calculated values.

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
sims.py: The main Flask application. It handles user input, performs the modulation simulations, calculates metrics, and serves the results and charts to the web page.
modulation.py: Contains functions related to modulation, including mapping bit strings to constellation points.
ber.py: Contains functions for calculating the Bit Error Rate (BER) for different modulation schemes.
templates/index.html: The HTML template that defines the user interface, including the input form and the Chart.js-based charts for visualizing the simulation data.
Running the Application
Open a terminal and navigate to the project directory.

Run the Flask application with:

 bash 
python sims.py
Open your web browser and go to http://localhost:5000.

How to Use
Input Section
Type of Digital Modulation: Choose between QAM, PSK, FSK, ASK, or CSS using the radio buttons. The label and buttons are aligned to the left.
Modulation Order (2^x): Select the modulation order (e.g., 2, 4, 8, etc.). The number of bits per symbol is determined by the logarithm base 2 of the modulation order.
SNR (dB): Enter the signal-to-noise ratio in dB.
Target Data Rate (Mbps): Input the desired throughput in Mbps.
Sample Data Stream (bits): Provide a bit stream (a string of 0s and 1s). The length must be a multiple of the number of bits per symbol (determined by the modulation order).
Auto Compute: Check this option to enable automatic computation when inputs change.
Click Run Simulation to start the simulation.
Result Section
This section displays:

Calculated Value: Simulation parameters such as SNR requirement, bandwidth, bits per symbol, baud rate, and effective throughput.
Standard Value: Standard values for the selected modulation scheme and order, allowing for comparison.
Chart Section
The charts are divided as follows:

2×2 Grid:
Time Domain Signal: Plot of the generated time-domain waveform.
Frequency Spectrum: The power spectral density (in dB/Hz) estimated using Welch’s method.
BER vs. SNR: A plot showing the relationship between SNR and Bit Error Rate.
SNR vs. Throughput: A line chart showing how throughput varies with SNR.
Full-width Charts:
Constellation Map: Displays all possible constellation points along with the actual input symbols (highlighted). Cross lines indicate the in-phase and quadrature axes.
Noise Tolerance Map: Displays circles (tolerance boundaries) around each constellation point to indicate possible noise tolerance.
Customization
Simulation Logic: Modify the simulation formulas or parameters in sims.py, modulation.py, and ber.py as needed.
Chart Appearance: Adjust the Chart.js configuration in templates/index.html to customize chart appearance, axis limits, or other settings.
Noise Tolerance and Constellation Mapping: Further refine the noise tolerance and constellation mapping logic to suit your simulation requirements.
Troubleshooting
Charts Not Displaying:
Open your browser’s developer console to check for any errors.
Ensure that the backend is providing non-empty data arrays for each chart.
Input Issues:
Verify that the sample data stream meets the requirement (length must be a multiple of bits per symbol).
FFT/PSD Problems:
Ensure that enough time-domain samples are generated for the FFT used in Welch’s method.
License
GNU GENERAL PUBLIC LICENSE Version 3

This project is provided "as is" without any warranty.

Created by Engr. Dennis A. Garcia