# 3D Scene Reconstruction and Virtual Tour - SFM Project

This project implements a **Structure-from-Motion (SfM) pipeline** to reconstruct a 3D scene from a sequence of 2D images, culminating in the visualization of a dense **Point Cloud** via an interactive **Virtual Tour**.

---

## Key Features

* **Two-Phase Pipeline:** Modular implementation covering two-view and multi-view SfM.
* **Point Cloud Generation:** Successfully generates and exports the final 3D point cloud.
* **Web Visualization:** Interactive virtual tour accessible via a standard HTTP server.

---

## Repository Structure

The core files and directories for the project are organized as follows:
├── data/ # Source images and configuration files 
├── outputs/ # Intermediate and final point cloud files/data 
├── virtual_tour/ # Web assets (HTML, JS, CSS) for the final visualization 
├── phase1_two_view_sfm.py # Script for the initial two-view reconstruction (Phase 1) 
├── phase2_multiview_sfm.py # Script for the incremental multi-view SfM (Phase 2) 
├── report.pdf # Detailed technical report of the project
├── AI_Usage.md # Documentation regarding the use of AI assistance 
└── .gitignore # Files/directories to be ignored by Git

---

## Getting Started

To run the entire pipeline and view the final result, follow these two main steps.

### 1. Run the Structure-from-Motion Pipeline

Execute the two processing phases sequentially to generate the point cloud data, which will be saved in the `outputs/` directory.

| Phase | Script Name | Command |
| :--- | :--- | :--- |
| **Phase 1** | `phase1_two_view_sfm.py` | `python phase1_two_view_sfm.py` |
| **Phase 2** | `phase2_multiview_sfm.py` | `python phase2_multiview_sfm.py` |

### 2. Launch the Virtual Tour

Once the point cloud is generated, you can launch the visualization using a Python simple HTTP server.

1.  **Change Directory:** Navigate into the visualization folder:
    ```bash
    cd web_viewer
    ```

2.  **Start Server:** Launch a Python simple HTTP server on port 8000:
    ```bash
    python -m http.server 8000
    ```

3.  **Access Visualization:** Open your web browser and navigate to the local host address:
    ```
    http://localhost:8000
    ```

4.  **View Tour:** Click the link labeled **"virtual tour"** (or the main HTML file) to load the 3D visualization.
