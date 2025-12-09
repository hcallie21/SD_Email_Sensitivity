# SD_Email_Sensitivity
A privacy-focused Chrome Extension that detects and highlights sensitive information (PII) in your emails in real-time, powered by a local TinyBERT AI model.

## Features
-   **Real-time Detection**: Scans email drafts as you type
-   **Hybrid Intelligence**: 
    -   **TinyBERT AI**: Understands context to detect sensitive statements.
-   **Smart Filtering**: Automatically ignores safe greetings ("Hi", "Dear", "Thanks") to prevent false alarms.
-   **Privacy First**: All processing happens on your local machine via a Python server. No data leaves your computer.

## AI Model Engine
- This project runs on **TinyBERT.** 
- **Why TinyBERT?** It is 96% smaller and 4x faster than standard BERT, making it perfect for real-time analysis on laptops without needing a powerful GPU.

## Installation

### 1. Start the Local Server
The backend requires Python to run the AI model.

1.  Navigate to the `model` directory:
    ```bash
    cd model
    ```
2.  Install dependencies (if not already installed):
    ```bash
    pip install flask flask-cors torch transformers
    ```
3.  Run the server:
    ```bash
    python server.py

    ```
    *The server will start on http://127.0.0.1:5000*

### 2. Load the Chrome Extension
1.  Open Chrome and go to `chrome://extensions`.
2.  Enable **Developer Mode** (top right toggle).
3.  Click **Load unpacked**.
4.  Select the `extension` folder from this project.

## Usage

1.  Open **Gmail** (or any webmail client).
2.  Start composing a new email.
3.  Type or Paste text.
    -   *Example*: "My name is John. sending my credit card..."
4.  The extension will highlight sensitive sentences in **Red**.
5.  A shield icon in the bottom right will show the total count of sensitive items found.
## Customization

You can adjust sensitivity settings in `model/server.py`:
-   `OVERALL_THRESHOLD`: Confidence score required for AI detection (Default: 0.90).
