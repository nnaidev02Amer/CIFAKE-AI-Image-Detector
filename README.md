# CIFAKE – AI Image Detector

An academic AI-powered image classification project that identifies whether an image is **REAL (authentic)** or **FAKE (AI-generated)** using deep learning.

The project uses a custom Convolutional Neural Network (CNN) as its primary classifier. It also includes an experimental MobileNetV3-Small model for comparison and an optional Google Gemini integration that provides a separate visual second opinion.

The application is built with **Python, PyTorch, Streamlit, and Google Gemini API**.

---

## 1. Project Overview

AI-generated images are becoming increasingly realistic. Distinguishing between real photographs and AI-generated images is an important challenge in computer vision.

This project provides an interactive web application where users can upload images and examine the model's predictions.

### Main Features

- **CNN Image Detection:** Classifies uploaded images as REAL or FAKE.
- **Prediction Score:** Displays the model's predicted probability of REAL.
- **Explainable AI:** Uses Grad-CAM to highlight image regions that influenced the CNN prediction.
- **Model Evaluation:** Displays evaluation metrics and model performance.
- **Experimental Comparison:** Compares the original CNN with an experimental MobileNetV3-Small model.
- **Gemini AI Analysis:** Sends an image to Google Gemini only after the user acknowledges the privacy notice and requests analysis.
- **Image Validation:** Supports JPG, JPEG, PNG, and WEBP images, subject to the application's upload validation.
- **Error Handling:** Provides clear messages for invalid images, missing API keys, API errors, and unavailable services.
- **Streamlit Interface:** Provides an interactive browser-based dashboard.

> **Important:** This is an academic prototype, not a guaranteed AI-image detector or an image-provenance verification system.

---

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Image transformations and pretrained model support |
| CNN | Primary image classification model |
| MobileNetV3-Small | Experimental comparison model |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning evaluation metrics |
| Pillow (PIL) | Image loading and processing |
| Matplotlib | Evaluation charts and visualizations |
| Grad-CAM | Explainable AI visualization |
| Google Gemini API | Optional image analysis and second opinion |
| python-dotenv | Loading environment variables from `.env` |
| pytest | Automated testing |
| Git and GitHub | Version control and project hosting |

The exact dependency versions are maintained in `requirements.txt`.

---

## 3. Application Pages

The application contains seven pages.

### 3.1 Dashboard

The main dashboard provides an overview of the project, available functionality, and model information.

### 3.2 Image Detection

This is the primary image classification page.

Steps:
1. Upload an image.
2. The application validates the image.
3. The image is preprocessed for the CNN.
4. The trained CNN predicts REAL or FAKE.
5. The application displays the prediction and its P(REAL) score.

The original CNN uses a 32 × 32 input image.

### 3.3 Explainable AI

This page provides a Grad-CAM visualization to help explain which image regions influenced the CNN prediction.

Grad-CAM is an explanatory visualization. It does not prove that an image is real or AI-generated.

### 3.4 Model Evaluation

Displays the original CNN's evaluation information, such as:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Other available evaluation charts

### 3.5 About

Provides information about the project, its objective, methodology, and technologies.

### 3.6 Experimental Comparison

Compares the original CNN against an experimental MobileNetV3-Small model.

The experimental model was trained on a smaller local CIFAKE subset. It is not a like-for-like comparison with the original CNN, which was trained using the full training dataset.

### 3.7 Gemini AI Analysis

Provides a second opinion using Google's Gemini API.

The page:
- Reuses the image uploaded on the Image Detection page.
- Displays the original CNN prediction as the primary result.
- Sends a separate metadata-stripped image copy to Gemini.
- Displays Gemini's assessment, explanation, evidence, and limitations.
- Shows a comparison state between the two systems.
- Requires the user to acknowledge the privacy notice and click the analysis button.

Gemini responses use three assessment categories:

- `POSSIBLY_AI_GENERATED`
- `POSSIBLY_REAL`
- `INCONCLUSIVE`

The application does not combine CNN and Gemini predictions into one score.

---

## 4. Project Architecture

The application follows a modular structure.

### CNN Detection Workflow

1. The user uploads an image.
2. The application validates the image.
3. The image is converted into RGB format where necessary.
4. The CNN preprocessing pipeline resizes the image to 32 × 32.
5. The trained model performs inference.
6. The application displays the predicted class and P(REAL).

### Gemini Analysis Workflow

1. The user uploads an image on Image Detection.
2. The application retains the original uploaded bytes for the current session.
3. The user opens Gemini AI Analysis.
4. The privacy notice and acknowledgement are displayed.
5. The user explicitly clicks **Send to Gemini**.
6. The application prepares a separate image copy without EXIF, GPS, XMP, IPTC, and comment metadata.
7. The image and analysis instructions are sent to Gemini.
8. Gemini returns a structured response.
9. The application validates the response and displays the assessment.
10. The result is retained for that image to avoid unnecessary repeat requests when switching pages.

The original uploaded image and the CNN processing pipeline are not changed by Gemini metadata removal.

### Gemini Image Processing Notes

| Upload type | Processing before Gemini | Image quality |
|---|---|---|
| Normal JPEG | Metadata removed without recompression | Pixel data remains identical |
| JPEG with rotation flag | Rotated upright and re-saved at quality 95 | Some additional compression |
| PNG | Re-saved losslessly | Pixel data remains identical |
| WEBP | Re-saved losslessly | Pixel data remains identical |

The rotation case may introduce a small image-quality change, which is documented in the application.

---

## 5. Dataset

The project uses the CIFAKE dataset.

The dataset contains two classes:

- **REAL:** Real-world images.
- **FAKE:** AI-generated images.

### Dataset Structure

The expected dataset structure is:

```text
data/
├── train/
│   ├── REAL/
│   └── FAKE/
│
└── test/
    ├── REAL/
    └── FAKE/
```

The reported dataset counts are:

| Dataset split | REAL | FAKE | Total |
|---|---:|---:|---:|
| Training | 50,000 | 50,000 | 100,000 |
| Testing | 10,000 | 10,000 | 20,000 |
| Total | 60,000 | 60,000 | 120,000 |

The project expects the dataset to be available locally. Do not assume that cloning the repository automatically downloads the dataset.

### Dataset Notes

- Keep the original dataset structure unchanged.
- Do not mix training and test images.
- Use known labels when evaluating the model.
- Avoid using the same images in both training and testing.
- External images should be treated as generalisation tests, not as part of the original CIFAKE test metrics.

---

## 6. Original CNN Model

The original CNN is the primary classification model.

### Model Information

| Property | Value |
|---|---|
| Model type | Custom Convolutional Neural Network |
| Input dimensions | 32 × 32 |
| Parameters | 634,753 |
| Loss function | Binary Cross Entropy |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Training epochs | 10 |
| Output | REAL / FAKE |
| Checkpoint | `models/cifake_cnn.pth` |

### Architecture Overview

The original architecture uses:
- Three convolutional blocks.
- Batch normalization.
- ReLU activation.
- Max pooling.
- Dropout.
- Fully connected classification layers.
- Sigmoid output.

The model produces a probability used by the application to determine its displayed prediction.

### Reported Original CNN Results

The following results were reported from evaluation on the full 20,000-image CIFAKE test set.

| Metric | Reported result |
|---|---:|
| Accuracy | 93.455% |
| ROC-AUC | 0.9921 |
| FAKE precision | 98.34% |
| FAKE recall | 88.40% |
| Brier score | 0.0508 |

These are reported CIFAKE test-set results, not guaranteed performance on images from other datasets or newly generated AI images.

### Original CNN Confusion Matrix

| Actual / Predicted | Predicted REAL | Predicted FAKE |
|---|---:|---:|
| Actual REAL | 9,851 | 149 |
| Actual FAKE | 1,160 | 8,840 |

---

## 7. Experimental MobileNetV3-Small Model

An additional MobileNetV3-Small experiment was implemented to explore an alternative architecture.

### Experiment Configuration

| Property | Value |
|---|---|
| Model | MobileNetV3-Small |
| Pretrained weights | ImageNet |
| Input size | 128 × 128 |
| Training subset | 800 REAL + 800 FAKE |
| Validation subset | 100 REAL + 100 FAKE |
| Test subset | 100 REAL + 100 FAKE |
| Training epochs planned | 3 |
| Batch size | 32 |
| Augmentation | Horizontal flip |
| Hardware | CPU |

### Reported Comparison Results

Both models were evaluated on the same 200-image experimental test sample.

| Metric | Original CNN | MobileNetV3-Small |
|---|---:|---:|
| Accuracy | 93.0% | 77.5% |
| FAKE precision | 96.7% | 92.3% |
| FAKE recall | 89.0% | 60.0% |
| FAKE F1-score | 0.927 | 0.727 |
| ROC-AUC | 0.994 | 0.926 |
| False positive rate | 3% | 5% |
| False negative rate | 11% | 40% |
| Errors | 14 | 45 |
| CPU inference time | 3–4 ms | Approximately 23 ms |

### Important Experimental Limitation

This is not a like-for-like training comparison.

The original CNN was trained using the full CIFAKE training dataset, while MobileNetV3-Small was trained using a small subset.

The 200-image experiment is intended for academic demonstration and comparison. It does not establish that the original CNN is universally better than MobileNetV3-Small.

The original CNN remains the primary model.

---

## 8. Gemini API Integration

The Gemini integration is optional.

It provides a separate visual assessment of an uploaded image. Gemini is not treated as a verified image-provenance detector.

### Requirements

- A Google account eligible to use Google AI Studio.
- A Gemini API key.
- Internet connectivity.
- The `google-genai` package.
- The `python-dotenv` package.

### Create a Gemini API Key

1. Open Google AI Studio:

   https://aistudio.google.com/

2. Select **Get API key**.
3. Create or select a Google Cloud project as prompted.
4. Generate an API key.
5. Copy the key.
6. Store it locally in the project's `.env` file.

Never upload your real API key to GitHub or share it in screenshots, source code, or chat messages.

### Configure the API Key

Create a `.env` file in the project root.

Copy the example:

```powershell
Copy-Item .env.example .env
```

Open the file:

```powershell
notepad .env
```

Add your configuration:

```env
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Replace the placeholder with your own API key.

The model can be changed through `GEMINI_MODEL` if another supported model is required.

### API Key Configuration Methods

The application can read the API key from:
1. Environment variables.
2. A local `.env` file.
3. Streamlit secrets.

The key must not be hardcoded in Python source files.

### Verify `.env` Is Ignored by Git

Run:

```powershell
git check-ignore .env
```

The expected output is:

```text
.env
```

The `.env.example` file is safe to commit because it contains placeholders only.

### Gemini Privacy and Cost

Before sending an image:
- Read the privacy notice in the application.
- Use non-personal images for demonstrations.
- Understand Google's applicable API data-use terms.
- Check the current pricing and quota for your account.

Free-tier and paid-tier data-use terms can differ. Do not send sensitive or confidential images.

Gemini's accuracy for this specific classification task has not been established by a labelled evaluation in this project.

---

## 9. System Requirements

Recommended environment for running the project:

| Component | Requirement |
|---|---|
| Operating system | Windows 10 / Windows 11 |
| Python | A compatible Python version for the installed PyTorch and other dependencies |
| RAM | 8 GB or more recommended |
| Storage | Sufficient space for the dataset, model checkpoints, and Python packages |
| Processor | Modern multi-core CPU |
| GPU | Optional |
| Internet | Required for Gemini API; also required for package installation and pretrained-weight downloads when needed |
| Browser | Chrome, Edge, Firefox, or another modern browser |

The original CNN can perform inference on CPU.

The experimental MobileNet model and any training operation may require more time on CPU.

For reproducibility, use the Python environment and package versions recorded in your project setup.

---

## 10. Installation Guide – Windows

Follow these instructions from the project root folder.

### Step 1: Install Python

Install a Python version compatible with the packages in `requirements.txt`.

Check your installation:

```powershell
python --version
```

Also check pip:

```powershell
python -m pip --version
```

### Step 2: Clone the Repository

```powershell
git clone https://github.com/sofiyann-codes/CIFAKE-AI-Image-Detector.git
```

Open the project folder:

```powershell
cd CIFAKE-AI-Image-Detector
```

If you already cloned the project, navigate to your existing project directory instead.

### Step 3: Create a Virtual Environment

```powershell
python -m venv venv
```

### Step 4: Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 6: Install Project Dependencies

```powershell
pip install -r requirements.txt
```

This installs the packages listed in the project's dependency file, including the Gemini integration packages.

### Step 7: Verify Important Packages

```powershell
python -m pip show torch
python -m pip show torchvision
python -m pip show streamlit
python -m pip show google-genai
python -m pip show python-dotenv
python -m pip show scikit-learn
python -m pip show Pillow
```

If a package is missing, check the installation output and `requirements.txt`.

### Step 8: Verify Project Files

Check that the model checkpoint exists:

```powershell
Test-Path .\models\cifake_cnn.pth
```

Expected result:

```text
True
```

Check that the dataset directories exist:

```powershell
Test-Path .\data\train
Test-Path .\data\test
```

The dataset is required for dataset evaluation and training workflows. The application also needs the trained model checkpoint for CNN inference.

### Step 9: Configure Gemini

Follow Section 8 to create `.env` and add your API key.

If you do not want to use Gemini, you can leave the API key unconfigured and use the CNN functionality, provided the required application dependencies and model checkpoint are installed.

### Step 10: Run the Application

```powershell
streamlit run app/app.py
```

The application should open in your browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 11. How to Run the Project

### Start the Application

From the project root:

```powershell
streamlit run app/app.py
```

### Alternative Command

If the `streamlit` command is not recognized:

```powershell
python -m streamlit run app/app.py
```

### Stop the Application

Press:

```text
CTRL + C
```

in the terminal where Streamlit is running.

---

## 12. How to Use the Application

### A. Image Detection

1. Open the application.
2. Select **Image Detection**.
3. Upload a JPG, JPEG, PNG, or WEBP image.
4. Wait for the prediction.
5. View the predicted class and P(REAL) score.

### B. Explainable AI

1. Open **Explainable AI**.
2. Select or use an available image as supported by the page.
3. View the Grad-CAM heatmap.
4. Examine the highlighted regions.

### C. Experimental Comparison

1. Open **Experimental Comparison**.
2. Review the CNN and MobileNetV3-Small results.
3. Compare their reported performance.
4. Read the experiment limitations before drawing conclusions.

### D. Gemini AI Analysis

1. Open **Image Detection**.
2. Upload a non-personal image.
3. Wait for the CNN prediction.
4. Open **Gemini AI Analysis**.
5. Read the privacy notice.
6. Tick the acknowledgement checkbox.
7. Click **Send to Gemini**.
8. Wait for the API response.
9. Review the Gemini assessment and compare it with the CNN prediction.

Gemini does not automatically run when an image is uploaded. The user must explicitly request analysis.

---

## 13. Supported Image Formats and Validation

The image upload functionality supports:

| Format | Extension |
|---|---|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| WEBP | `.webp` |

The existing image validator includes:
- File-type validation.
- Image decoding validation.
- Empty and corrupt image handling.
- Upload-size validation.
- RGB conversion where necessary.
- Protection against decompression-bomb images.

The reported upload size limit is 10 MB.

Some multi-picture JPEG/MPO files are not supported by the current upload validator.

---

## 14. Project Folder Structure

The following is the main project structure. Some generated result files and individual experiment artifacts may vary depending on the current checkout.

```text
CIFAKE-AI-Image-Detector/
│
├── app/
│   ├── app.py
│   ├── model.py
│   ├── gradcam.py
│   ├── image_utils.py
│   ├── experimental_page.py
│   ├── gemini_analyzer.py
│   ├── gemini_page.py
│   └── ...
│
├── data/
│   ├── train/
│   │   ├── REAL/
│   │   └── FAKE/
│   │
│   └── test/
│       ├── REAL/
│       └── FAKE/
│
├── models/
│   ├── cifake_cnn.pth
│   │
│   └── experimental/
│       └── mobilenetv3_small_128.pth
│
├── experiments/
│   ├── common.py
│   ├── prepare_split.py
│   ├── train_mobilenet.py
│   ├── compare_models.py
│   ├── README.md
│   ├── DATA_SOURCES.md
│   ├── split_manifest.csv
│   └── ...
│
├── samples/
│   └── ...
│
├── scripts/
│   └── evaluate_model.py
│
├── tests/
│   ├── test_streamlit_app.py
│   ├── test_gemini_analyzer.py
│   └── ...
│
├── docs/
│   └── GEMINI_SETUP.md
│
├── outputs/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Some files and directories may not exist in every version of the repository. Refer to the actual project checkout for the complete file listing.

### Important Directories

| Directory | Purpose |
|---|---|
| `app/` | Streamlit application and model integration |
| `data/` | CIFAKE dataset |
| `models/` | Original and experimental checkpoints |
| `experiments/` | Experimental training, split preparation, and comparison |
| `samples/` | Sample images for testing |
| `scripts/` | Evaluation scripts |
| `tests/` | Automated tests |
| `docs/` | Additional documentation |
| `outputs/` | Generated evaluation outputs |
| `notebooks/` | Research and experimentation notebooks |

---

## 15. Dependencies and Python Packages

The complete dependency list is maintained in:

```text
requirements.txt
```

Install all dependencies using:

```powershell
pip install -r requirements.txt
```

### Important Packages

| Package | Purpose |
|---|---|
| `torch` | CNN model loading and inference |
| `torchvision` | Image transformations and MobileNet support |
| `streamlit` | Web application |
| `numpy` | Numerical processing |
| `scikit-learn` | Evaluation metrics |
| `Pillow` | Image processing |
| `matplotlib` | Graphs and evaluation visualizations |
| `google-genai` | Google Gemini API integration |
| `python-dotenv` | Loading `.env` configuration |
| `pytest` | Running automated tests |

The table describes the major packages. It is not a replacement for `requirements.txt`, which is the authoritative installation list.

### Install Gemini Packages Separately

If you only need to install the Gemini dependencies:

```powershell
pip install "google-genai>=3.4,<4" "python-dotenv>=1.0"
```

Prefer installing the complete requirements file when setting up a fresh environment.

---

## 16. Testing

The project includes automated tests for the application, model utilities, image validation, and Gemini integration.

Run the complete test suite:

```powershell
python -m pytest -q
```

The reported verification result after Gemini integration was:

```text
80 passed
```

This included:
- 34 previously existing tests.
- An additional Streamlit page test.
- 45 Gemini integration tests.

The tests use a fake Gemini client where appropriate, so the test suite does not need to make real Gemini API requests.

### Run Specific Tests

For Gemini:

```powershell
python -m pytest tests/test_gemini_analyzer.py -v
```

For the Streamlit application:

```powershell
python -m pytest tests/test_streamlit_app.py -v
```

Run the complete suite after changing the code.

---

## 17. Model Evaluation

The original CNN was evaluated on the CIFAKE test dataset.

The evaluation script is:

```text
scripts/evaluate_model.py
```

Run it from the project root:

```powershell
python scripts/evaluate_model.py
```

Check the script's available arguments and the existing evaluation documentation before using optional parameters.

### Evaluation Metrics

**Accuracy**

The proportion of all predictions that are correct.

**Precision**

The proportion of predicted positive cases that are correct.

**Recall**

The proportion of actual positive cases correctly identified.

**F1-score**

The harmonic mean of precision and recall.

**ROC-AUC**

Measures the model's ability to distinguish between the two classes across classification thresholds.

**Confusion Matrix**

Shows correct and incorrect predictions for both classes.

**Brier Score**

Measures the accuracy of probabilistic predictions.

For this project, the positive-class convention used in the reported confusion matrix is FAKE.

---

## 18. Limitations

This project has several important limitations.

### 18.1 Dataset Generalisation

The original CNN was trained and evaluated on CIFAKE. It may not perform reliably on images from unrelated sources or newer image-generation systems.

### 18.2 Low Resolution

The original CNN resizes images to 32 × 32. Fine visual details may be lost during preprocessing.

### 18.3 False Predictions

A real image may be classified as FAKE, and an AI-generated image may be classified as REAL.

A high P(REAL) score is the model's output, not proof of authenticity.

### 18.4 External AI Images

An image generated using a newer AI system may not have the same visual characteristics as the images in the training dataset.

### 18.5 MobileNet Experimental Results

MobileNetV3-Small was trained using a smaller dataset subset. Its experimental results should not be treated as a direct comparison of architectures under equal training conditions.

### 18.6 Gemini Limitations

Gemini provides a visual assessment based on the submitted image. It does not verify an image's source, generation history, or cryptographic provenance.

Its accuracy for this project has not been established through a dedicated labelled benchmark.

### 18.7 Grad-CAM Limitations

Grad-CAM shows regions that influenced the CNN prediction. It does not establish that those regions contain definitive evidence of AI generation.

---

## 19. Troubleshooting

### Problem 1: Streamlit Command Not Found

Run:

```powershell
python -m streamlit run app/app.py
```

If Streamlit is not installed:

```powershell
pip install streamlit
```

### Problem 2: Module Not Found

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

Confirm that the virtual environment is activated.

### Problem 3: Model Checkpoint Not Found

Check that this file exists:

```text
models/cifake_cnn.pth
```

Do not rename or replace the original checkpoint unless the application configuration is updated accordingly.

### Problem 4: Dataset Not Found

Check:

```text
data/train/
data/test/
```

Ensure the REAL and FAKE directories are present.

### Problem 5: Gemini API Key Missing

Check that `.env` exists in the project root and contains:

```env
GEMINI_API_KEY=your_actual_api_key
```

Restart Streamlit after changing environment configuration.

### Problem 6: Gemini API Error

Possible causes:
- Invalid API key.
- Quota exceeded.
- Model unavailable.
- Network connection failure.
- Request timeout.
- Safety restriction.

Read the error message shown in the application. The CNN result should remain available even when Gemini fails.

### Problem 7: Invalid Image

Try a standard JPG, PNG, or WEBP file.

Check:
- The file opens correctly.
- The file is not empty.
- The file is within the 10 MB upload limit.
- The file is not an unsupported MPO image.

### Problem 8: PowerShell Does Not Activate the Virtual Environment

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Problem 9: Port 8501 Is Already in Use

Run the application on another port:

```powershell
streamlit run app/app.py --server.port 8502
```

Open:

```text
http://localhost:8502
```

---

## 20. Git and GitHub Commands

Check the current Git status:

```powershell
git status
```

Check whitespace and patch errors:

```powershell
git diff --check
```

Review changed files:

```powershell
git diff --stat
```

Before committing, verify that:
- `.env` is ignored.
- `.streamlit/secrets.toml` is ignored.
- API keys are not present in source files.
- Large dataset files are handled according to the repository's existing Git configuration.
- Model checkpoints are included or excluded intentionally.

### Example Commit Workflow

```powershell
git add README.md app/ tests/ docs/ requirements.txt .gitignore
git status
git diff --cached --check
git commit -m "Document CIFAKE AI image detector and Gemini integration"
```

Only push after reviewing the staged files and confirming that no secrets or unintended dataset changes are included.

---

## 21. Future Improvements

Potential future work includes:

- Evaluate the CNN on additional independently labelled datasets.
- Test images from different AI image-generation systems.
- Increase image resolution and study its impact on classification.
- Compare architectures using equal training data and evaluation conditions.
- Improve robustness against resizing, compression, and blur.
- Build a larger labelled benchmark for Gemini assessment.
- Improve model calibration and uncertainty reporting.
- Explore image provenance standards and metadata-based verification.
- Add more comprehensive experiment tracking.

These are possible improvements, not features claimed to be completed.

---

## 22. Academic Project Summary

**Project Title:** CIFAKE – AI Image Detector

**Project Category:** Artificial Intelligence / Deep Learning / Computer Vision

**Primary Objective:** To develop an interactive application that classifies images as REAL or FAKE using a trained CNN model.

**Primary Model:** Custom Convolutional Neural Network.

**Experimental Model:** MobileNetV3-Small.

**Explainability Method:** Grad-CAM.

**Optional AI Integration:** Google Gemini API.

**Application Framework:** Streamlit.

**Programming Language:** Python.

### Project Outcome

The project demonstrates:
- Deep learning-based binary image classification.
- Image preprocessing and validation.
- Model evaluation using standard classification metrics.
- Explainable AI using Grad-CAM.
- Experimental comparison of deep learning models.
- Optional integration of an external generative AI API.
- Interactive deployment through a local Streamlit application.

The project is intended for academic learning, experimentation, and demonstration. It should not be used as the sole basis for deciding whether an image is authentic.

---

## 23. References and Resources

- **CIFAKE Dataset:** https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- **PyTorch:** https://pytorch.org/
- **Torchvision:** https://pytorch.org/vision/
- **Streamlit:** https://streamlit.io/
- **Scikit-learn:** https://scikit-learn.org/
- **Google AI Studio:** https://aistudio.google.com/
- **Gemini API Documentation:** https://ai.google.dev/gemini-api/docs
- **Google Gen AI SDK:** https://github.com/googleapis/python-genai
- **Python:** https://www.python.org/
- **Grad-CAM:** https://arxiv.org/abs/1610.02391

---

## 24. License and Disclaimer

Check the original repository's license and the dataset's terms before redistributing project code, model weights, or dataset files.

This project is developed for educational and academic purposes.

The predictions are model estimates and may be incorrect. Neither the CNN nor Gemini can guarantee the authenticity or provenance of an image.

**Always interpret the output as an AI-assisted assessment, not definitive proof.**
