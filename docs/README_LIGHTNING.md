# Running on Lightning AI Studios

This guide explains how to set up and run the Football Event Detection system on Lightning AI.

## Prerequisites

1.  **Lightning AI Account**: Sign up at [lightning.ai](https://lightning.ai).
2.  **GPU Instance**: Create a new Studio and select a GPU.
    - **Recommended**: NVIDIA A10G or A100.
    - **Note**: Modals like Molmo-7B and Qwen2.5-VL require significant VRAM. 4-bit quantization is enabled to reduce this.

## Setup Instructions

1.  **Open a Terminal** in your Lightning AI Studio.
2.  **Upload the Code**: You can drag and drop the project folder or use `git clone`.
3.  **Run Setup Script**:
    ```bash
    bash setup_lightning.sh
    ```
    This script will install all necessary libraries, including specialized ones like `qwen-vl-utils` and `flash-attn`.

4.  **Upload YOLO Models**:
    Ensure the following files are placed in the `data/` directory:
    - `data/football-player-detection.pt`
    - `data/football-ball-detection.pt`
    - `data/football-pitch-detection.pt`

5.  **Add Input Video**:
    Place your `.mp4` video files in the `input_video/` directory.

## Running the Analysis

To start the detection and analysis process:

```bash
python main4.py
```

### Outputs
- **JSON Results**: `match_analysis_result.json`
- **CSV Results**: `shots_report.csv` (if shot detection is active)
- **HTML Report**: `scout_report.html` (view this in your browser)
- **Annotated Video**: Found in the output directory if `SAVE_ANNOTATED_VIDEO` is set to `True`.

## Troubleshooting

- **OOM (Out Of Memory)**: If you encounter CUDA OOM errors, ensure you are using a GPU with at least 24GB VRAM.
- **Missing Modules**: If any module is missing after setup, try `pip install <module_name>`.
