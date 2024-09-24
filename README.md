# CV Tutorial - Object Detection Workshop

Welcome to the **Object Detection Workshop**! This repository contains materials and code for an introductory workshop on object detection, covering both fundamentals and coding examples. By following the setup instructions below, you can get started with exploring object detection techniques in real-time and batch inference scenarios.

In this workshop we will go through:

- Fundamentals of object detection
- When to do object detection and its types
- Types
- Coding
- Model examples
- Architecture

---

## Setup Instructions

### Step 1: Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/Thivyesh/cv-tutorial-cap.git
cd cv-tutorial-cap
```

### Step 2: Create a Virtual Environment

#### For Windows:
```bash
python3 -m venv cvw
cvw\Scripts\activate
```
#### For Mac/Linux:
```bash
python3 -m venv cvw
source cvw/bin/activate
```
### Step 3: Install Requirements

#### To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### Optional: Install Requirements Using Poetry

If you prefer to use Poetry for dependency management, follow these steps:

Activate the poetry shell:

```bash
poetry shell
```

Install dependencies from pyproject.toml:
Poetry uses the pyproject.toml file to manage dependencies. To install them, run:

```bash
poetry install
```

### Running the Object Detection Examples

This repository contains scripts for both real-time and batch object detection:

	•	Real-Time Object Detection: Uses your webcam for live object detection. These scripts are located in the rt/ directory.
	•	Batch Object Detection: Processes multiple images in a batch. These scripts are located in the batch/ directory.

### Folder Structure

	•	od/: Contains object detection-related scripts.
	•	rt/: Real-time object detection examples.
	•	batch/: Scripts for batch processing.
	•	ic/: Introductory content and examples related to the workshop.

### Real time and batch inference
The scripts are located under the folder "od". 
The "rt" folder contains scripts for running realtime inference on you're webcam.
The "batch" folder contains scripts for running batch inference.

### About Object Detection

Object detection is a core task in computer vision, involving both the identification and localization of objects within an image or video frame. The workshop covers both the real-time and batch processing approaches for various real-world applications, such as autonomous vehicles, surveillance, and more.

### Real-Time Object Detection

Real-time object detection refers to the ability of a system to perform object detection tasks with minimal delay, allowing for immediate response to changes in the environment. This capability is crucial in applications such as autonomous driving, live video analysis, and augmented reality, where timely detection of objects is essential for decision-making.

### Batch Object Detection

Batch object detection involves processing multiple images or video frames simultaneously to detect objects. Instead of processing each image or frame individually, batch processing allows for efficient utilization of computational resources and can significantly speed up the detection process. Batch object detection is commonly used in scenarios where large volumes of data need to be analyzed, such as video surveillance systems, satellite imagery analysis, and industrial quality control.

## Zero-Shot Learning in Object Detection

Zero-shot learning (ZSL) is a machine learning paradigm that aims to address the limitations of traditional object detection by enabling models to recognize objects without explicit training on them. Instead of relying solely on labeled examples of objects seen during training, ZSL models can generalize to recognize unseen objects based on their semantic descriptions or attributes.

In the context of object detection, ZSL allows a model to detect objects for which it has not been explicitly trained. This is achieved by leveraging semantic embeddings or attribute representations of objects, which capture high-level semantic information about their visual appearance, characteristics, and relationships.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

You can copy and paste the above Markdown into your `README.md` file directly. Let me know if you need further adjustments!

