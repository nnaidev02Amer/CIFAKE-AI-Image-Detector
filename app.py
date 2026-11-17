"""CIFAKE - AI Image Detector (Streamlit academic prototype).

Run from the project root OR from app/:
    streamlit run app/app.py        (root)
    streamlit run app.py            (inside app/)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from gradcam import compute_gradcam, render_heatmap, render_overlay  # noqa: E402
from image_utils import (ALLOWED_EXTENSIONS, MAX_FILE_BYTES,  # noqa: E402
                         ImageValidationError, load_image, preprocess, to_display_32)
from model import MODEL_PATH, PROJECT_ROOT, load_model, predict  # noqa: E402
import experimental_page  # noqa: E402  (Phase 2, reads saved results only)
import gemini_page  # noqa: E402  (optional Gemini second opinion)

try:  # optional: read GEMINI_API_KEY from a local .env (never overrides real env vars)
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

EVAL_PATH = PROJECT_ROOT / "outputs" / "evaluation_results.json"
SAMPLES_DIR = PROJECT_ROOT / "samples"
LOW_CONFIDENCE = 0.70

st.set_page_config(page_title="CIFAKE - AI Image Detector", page_icon="", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 100%); color: #fff;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.result {padding: 1rem 1.2rem; border-radius: 12px; font-size: 1.25rem; font-weight: 600;}
.result.real {background: #e8f7ee; color: #14532d; border: 1px solid #86d3a3;}
.result.fake {background: #fdecec; color: #7f1d1d; border: 1px solid #f0a3a3;}
.small {font-size: .88rem; opacity: .85;}
</style>
""", unsafe_allow_html=True)


# ── Cached resources ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_data
def load_eval():
    if not EVAL_PATH.is_file():
        return None
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def analyse(data: bytes) -> dict:
    """Run validation, preprocessing, prediction and Grad-CAM for one image."""
    model = get_model()
    image = load_image(data)
    tensor = preprocess(image)
    pred = predict(model, tensor)
    cam = compute_gradcam(model, tensor, target_class=pred["label"])
    small = to_display_32(image)
    return {"image": image, "small": small, "pred": pred, "cam": cam, "original_bytes": data,
            "heatmap": render_heatmap(cam.cam), "overlay": render_overlay(small, cam.cam)}


def confidence_note(pred: dict) -> tuple[str, str]:
    if pred["confidence"] < LOW_CONFIDENCE:
        return "warning", ("Low confidence: the score is close to the decision boundary, "
                           "so treat this prediction as uncertain.")
    return "info", ("The model is fairly confident, but confidence is not proof: it only "
                    "reflects how similar the image is to CIFAKE training data.")


def limitation_warning():
    st.warning(
        "**Limitation:** this model was trained on low-resolution (32x32) CIFAKE images "
        "and has not been validated for real-world photos or other AI generators. "
        "Uploaded images are resized to 32x32, which removes most fine detail. "
        "Treat results as a demonstration, not as evidence.")


# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title(" CIFAKE Detector")
page = st.sidebar.radio("Navigate", ["Dashboard", "Image Detection", "Explainable AI",
                                     "Model Evaluation", "About Project",
                                     "Experimental Comparison", "Gemini AI Analysis"])
st.sidebar.markdown("---")
st.sidebar.caption("B.Tech academic prototype. CPU-only inference.")

st.markdown("""
<div class="hero"><h1> CIFAKE - AI Image Detector</h1>
<p>Classify an image as <b>REAL</b> or <b>AI-generated (FAKE)</b> with a CNN, and see
which regions influenced the decision using Grad-CAM.</p></div>
""", unsafe_allow_html=True)

try:
    get_model()
except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a traceback
    st.error(f"Could not load the model from `{MODEL_PATH}`: {exc}")
    st.stop()

# ── Dashboard ───────────────────────────────────────────────────────────────
if page == "Dashboard":
    ev = load_eval()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Custom CNN")
    c2.metric("Parameters", "634,753")
    c3.metric("Input size", "32 x 32 x 3")
    if ev:
        c4.metric("CIFAKE test accuracy", f"{ev['baseline']['accuracy']:.2%}")
    else:
        c4.metric("CIFAKE test accuracy", "not calculated")
    st.markdown("### Available features")
    st.markdown(
        "- **Image Detection** - upload JPG/PNG/WEBP and get REAL / FAKE with a score\n"
        "- **Explainable AI** - Grad-CAM heatmap and overlay for the predicted class\n"
        "- **Model Evaluation** - metrics calculated on the CIFAKE test set\n"
        "- **About Project** - problem, dataset, architecture and limitations")
    limitation_warning()

# ── Image Detection ─────────────────────────────────────────────────────────
elif page == "Image Detection":
    st.subheader("Upload an image")
    limitation_warning()
    uploaded = st.file_uploader(
        f"Drag and drop or browse (JPG, JPEG, PNG, WEBP, max {MAX_FILE_BYTES // 1024 // 1024} MB)",
        type=list(ALLOWED_EXTENSIONS))

    sample_names = sorted(p.name for p in SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.is_dir() else []
    sample = None
    if sample_names:
        with st.expander("...or try a sample image from the CIFAKE test set"):
            sample = st.selectbox("Sample", ["(none)"] + sample_names,
                                  help="Names show the true label (REAL / FAKE).")
    data, source_name = None, None
    if uploaded is not None:
        data, source_name = uploaded.getvalue(), uploaded.name
    elif sample and sample != "(none)":
        data, source_name = (SAMPLES_DIR / sample).read_bytes(), sample

    if data is None:
        st.info("Upload an image to begin.")
    else:
        key = hashlib.sha256(data).hexdigest()
        try:
            with st.spinner("Analysing image..."):
                if st.session_state.get("last_key") != key:
                    st.session_state["last"] = analyse(data)
                    st.session_state["last_key"] = key
                    st.session_state["last_name"] = source_name
            res = st.session_state["last"]
        except ImageValidationError as exc:
            st.session_state.pop("last", None)
            st.session_state.pop("last_key", None)
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("last", None)
            st.session_state.pop("last_key", None)
            st.error(f"Something went wrong while analysing the image: {exc}")
            st.stop()

        pred = res["pred"]
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Uploaded image**")
            st.image(res["image"], width="stretch")
            w, h = res["image"].size
            st.caption(f"{source_name} - {w}x{h}px, resized to 32x32 for the model")
        with right:
            css = "real" if pred["label"] == "REAL" else "fake"
            text = "REAL (authentic)" if pred["label"] == "REAL" else "FAKE (AI-generated)"
            icon = "✅" if pred["label"] == "REAL" else "🚨"
            st.markdown(f'<div class="result {css}">{icon} Prediction: {text}</div>',
                        unsafe_allow_html=True)
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{pred['confidence']:.1%}")
            m2.metric("Score P(REAL)", f"{pred['p_real']:.4f}")
            st.progress(float(pred["p_real"]), text="0 = FAKE  ...  1 = REAL")
            kind, msg = confidence_note(pred)
            getattr(st, kind)(msg)
            st.caption("Go to **Explainable AI** in the sidebar to see the Grad-CAM view.")

# ── Explainable AI ──────────────────────────────────────────────────────────
elif page == "Explainable AI":
    st.subheader("Grad-CAM explanation")
    res = st.session_state.get("last")
    if res is None:
        st.info("No image analysed yet. Go to **Image Detection** and upload an image first.")
    else:
        pred, cam = res["pred"], res["cam"]
        st.write(f"Explaining the predicted class: **{pred['label']}** "
                 f"(confidence {pred['confidence']:.1%}) for "
                 f"*{st.session_state.get('last_name', 'image')}*.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original (32x32 model input)**")
            st.image(res["small"].resize((256, 256)), width="stretch")
        with c2:
            st.markdown("**Grad-CAM heatmap**")
            st.image(res["heatmap"], width="stretch")
        with c3:
            st.markdown("**Overlay**")
            st.image(res["overlay"], width="stretch")
        if cam.blank:
            st.warning("The map is blank: no region produced positive evidence for the "
                       "predicted class, so no highlight is drawn.")
        st.markdown("#### How to read this")
        st.markdown(
            f"- Warm colours (red/yellow) mark regions that pushed the model **towards "
            f"{cam.target_class}**; blue regions contributed little.\n"
            "- The map comes from an 8x8 feature grid, so it is coarse by design.")
        st.info("Grad-CAM shows which regions influenced the model's decision. It does "
                "**not** prove that an image is authentic or AI-generated, and highlighted "
                "regions are not guaranteed to be actual generation artifacts.")

# ── Model Evaluation ────────────────────────────────────────────────────────
elif page == "Model Evaluation":
    st.subheader("Evaluation on the CIFAKE test set")
    ev = load_eval()
    if ev is None:
        st.warning("Evaluation results have not been calculated yet. Run "
                   "`python scripts/evaluate_model.py` (requires the dataset in `data/test`).")
    else:
        b = ev["baseline"]
        st.caption(f"Calculated on {ev['computed_on']} using {ev['test_images']:,} test images. "
                   "Precision, recall and F1 treat FAKE as the positive class.")
        a, p, r, f, u = st.columns(5)
        a.metric("Accuracy", f"{b['accuracy']:.2%}")
        p.metric("Precision", f"{b['fake_precision']:.2%}")
        r.metric("Recall", f"{b['fake_recall']:.2%}")
        f.metric("F1-score", f"{b['fake_f1']:.3f}")
        u.metric("AUC", f"{b['auc']:.4f}")

        cm = b["confusion_matrix"]
        matrix = np.array([[cm["true_fake_pred_fake"], cm["true_fake_pred_real"]],
                           [cm["true_real_pred_fake"], cm["true_real_pred_real"]]])
        st.markdown("**Confusion matrix** (rows = true label, columns = predicted)")
        st.dataframe(pd.DataFrame(matrix, index=["True FAKE", "True REAL"],
                                  columns=["Predicted FAKE", "Predicted REAL"]),
                     width="stretch")
        st.write(f"False positive rate (REAL flagged FAKE): **{b['false_positive_rate']:.2%}** - "
                 f"false negative rate (FAKE passed as REAL): **{b['false_negative_rate']:.2%}**")

        rob = ev.get("robustness")
        if rob:
            st.markdown("#### Robustness check (same test set, modified images)")
            rows = {"Original": b}
            rows["Gaussian blur (radius 0.6)"] = rob["gaussian_blur_radius_0.6"]
            rows["Upscale to 256 then resize to 32"] = rob["upscale_256_then_resize_32"]
            st.dataframe(pd.DataFrame(
                {"Accuracy": [f"{v['accuracy']:.2%}" for v in rows.values()],
                 "AUC": [f"{v['auc']:.4f}" for v in rows.values()],
                 "FAKE recall": [f"{v['fake_recall']:.2%}" for v in rows.values()]},
                index=list(rows)), width="stretch")
            st.warning("Accuracy drops sharply after mild blur or resampling, which suggests "
                       "the model relies on fragile low-level patterns specific to CIFAKE.")
        st.info("Known caveat: the CIFAKE train and test folders contain 378 byte-identical "
                "images in both, so the test result may be slightly optimistic. Only CIFAKE "
                "data has been evaluated; performance on other generators is unknown.")

# ── Experimental Comparison (Phase 2) ───────────────────────────────────────
elif page == "Experimental Comparison":
    experimental_page.render()

# ── Gemini AI Analysis (optional second opinion) ────────────────────────────
elif page == "Gemini AI Analysis":
    gemini_page.render()

# ── About ───────────────────────────────────────────────────────────────────
else:
    st.subheader("About the project")
    st.markdown("""
**Problem statement.** AI image generators produce pictures that are hard to tell apart from
photographs. This project explores whether a small CNN can separate real from
AI-generated images, and whether Grad-CAM can show what influenced its decision.

**Dataset.** CIFAKE (Bird & Lotfi, 2024): 60,000 REAL images from CIFAR-10 and 60,000 FAKE
images generated with Stable Diffusion, all 32x32 RGB. Split used here: 100,000 train
(50k/50k) and 20,000 test (10k/10k). No separate validation set.

**Model.** 3 x (Conv - BatchNorm - ReLU - MaxPool - Dropout) with 32/64/128 filters, followed by
Dense 256 - 64 - 1 with a Sigmoid. 634,753 parameters. Trained with BCE loss, Adam
(lr 0.001), batch size 64, 10 epochs.

**Technology.** Python, PyTorch, Torchvision, Streamlit, OpenCV, NumPy, Pillow,
Scikit-learn (evaluation), Matplotlib (notebooks).

**Limitations.**
- Trained and tested only on 32x32 CIFAKE images; real-world generalisation is unverified.
- Mild blur or resampling sharply reduces accuracy (see Model Evaluation).
- Only one generator (Stable Diffusion, as described by the CIFAKE dataset) is represented.
- Grad-CAM is a coarse 8x8 explanation, not proof of authenticity.

**Future scope.** Validation split and augmentation, evaluation on unseen generators, a fair
comparison with MobileNetV3 / EfficientNet-B0 / ResNet50, batch prediction, reports,
metadata inspection, ROC and training curves in the app. *(Proposed only - not implemented.)*
""")
