AI-Based Human Detection Under Debris (UWB Radar System)
📌 Project Overview
This project focuses on the detection of survivors trapped under earthquake rubble or structural debris using Ultra-Wideband (UWB) Radar technology. By leveraging low-frequency electromagnetic waves, the system can penetrate thick obstacles (up to 1.5m of concrete/debris) to detect the minute chest cavity movements associated with human breathing and heartbeats.

The system utilizes a Convolutional Neural Network (CNN) running on an NVIDIA Jetson Nano to classify signals in real-time, distinguishing between empty environments and the presence of a living human.

🚀 Key Features
Non-Contact Vital Sign Detection: Extracts breathing and heart rate frequencies from radar backscatter without physical contact.

Real-Time Signal Processing: Implements Bandpass Filtering (Scipy) and Fast Fourier Transforms (FFT) to isolate biological signals from environmental noise.

AI-Powered Classification: Uses a CNN-based approach to provide a "Confidence Score" for human presence.

Dynamic Live Dashboard: A comprehensive Matplotlib-based GUI that visualizes:

Raw UWB Signal: The unfiltered radar data.

Frequency Spectrum: Identifying peaks in the 0.1Hz–2.5Hz range (human vitals).

Extracted Vitals: Cleaned waveforms for breathing and pulse.

Detection Status: Visual alerts (Red for detection, Green for clear).

🛠️ Technical Stack
Language: Python 3.x

Signal Processing: NumPy, SciPy (Butterworth filters, FFT)

AI/ML Framework: TensorFlow/Keras (for the CNN model architecture)

Hardware Simulation: NVIDIA Jetson Nano integration logic.

Visualization: Matplotlib (Gridspec, Animation, and Patches)

🧬 Signal Processing Workflow
The project follows a rigorous pipeline to ensure accuracy in high-noise disaster environments:

Signal Acquisition: Receiving UWB pulses reflected from the target area.

Noise Reduction: Application of a 4th-order Butterworth bandpass filter to remove static clutter and high-frequency noise.

Feature Extraction: Conversion of time-domain signals into frequency-domain components to identify rhythmic biological patterns.

Inference: The CNN analyzes the variance and spectral density to confirm human life with high confidence.

📸 Simulation Preview
The simulation alternates between two phases to demonstrate the system's sensitivity:

Human Phase: Detects a 0.3Hz breathing rate and 1.2Hz heart rate through simulated debris.

Empty Phase: Baseline noise monitoring when no biological movement is present.

-Future Scope:
Multi-Person Detection: Using range-binning to identify multiple survivors at different depths.

Drone Integration: Mounting the UWB-Jetson payload on an autonomous UAV for rapid area scanning.

Thermal Fusion: Combining radar data with FLIR thermal imaging for multi-modal verification.
