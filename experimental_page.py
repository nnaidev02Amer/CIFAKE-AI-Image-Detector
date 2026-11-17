"""'Experimental Comparison' page (Phase 2).

Only READS saved results from experiments/results/ (produced by
experiments/compare_models.py). It does not load the experimental model, so the
rest of the app is unaffected and starts just as fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parent.parent / "experiments" / "results"


def _json(name: str):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _image(name: str, caption: str | None = None):
    path = RESULTS / name
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"`{name}` not found.")


def render() -> None:
    st.subheader("🧪 Experimental model comparison")
    st.error(
        "**EXPERIMENTAL - small demonstration evaluation.** These results come from one "
        "short training run on 1,600 CIFAKE images and are measured on only **200** test "
        "images. They are not definitive evidence that either model is better, and they say "
        "nothing about real-world photos or other AI generators. The app's predictions still "
        "use the original CNN.")

    cmp_, hist, split = (_json("comparison_results.json"), _json("training_history.json"),
                         _json("split_summary.json"))
    if cmp_ is None or hist is None:
        st.info("Experiment results not found. Run the three scripts in `experiments/` "
                "(see experiments/README.md).")
        return

    cnn, mnv3 = cmp_["models"]["cnn"], cmp_["models"]["mnv3"]

    st.markdown("#### 1. Same 200 held-out test images")
    st.caption("100 REAL + 100 FAKE from CIFAKE `data/test`; neither model trained on them, and "
               "none is an exact or near duplicate of any `data/train` image. FAKE is the "
               "positive class; threshold 0.5.")

    def row(m):
        return {"Accuracy": f"{m['accuracy']:.1%}",
                "Accuracy 95% CI": f"{m['accuracy_95ci'][0]:.1%} - {m['accuracy_95ci'][1]:.1%}",
                "Precision": f"{m['fake_precision']:.1%}", "Recall": f"{m['fake_recall']:.1%}",
                "F1": f"{m['fake_f1']:.3f}", "ROC-AUC": f"{m['auc']:.3f}",
                "FPR (REAL→FAKE)": f"{m['false_positive_rate']:.1%}",
                "FNR (FAKE→REAL)": f"{m['false_negative_rate']:.1%}",
                "Errors": f"{m['errors']} / {cmp_['test_images']}",
                "CPU ms / image": f"{m['inference']['single_image']['median_ms']:.1f}",
                "Parameters": f"{m['parameters']:,}"}

    st.dataframe(pd.DataFrame([row(cnn), row(mnv3)],
                              index=["Original CNN (32×32)", "EXPERIMENTAL MobileNetV3-Small (128×128)"]).T,
                 width="stretch")
    pt = cmp_["paired_test"]
    st.write(f"Images only the original CNN got right: **{pt['only_original_correct']}** · "
             f"only the experimental model got right: **{pt['only_experimental_correct']}** · "
             f"both wrong: **{pt['both_wrong']}** (exact McNemar p = {pt['mcnemar_exact_p_value']:.1e}).")
    better = "original CNN" if cnn["accuracy"] > mnv3["accuracy"] else "experimental model"
    st.info(f"On this test set the **{better}** scored higher. The original CNN was trained on "
            "100,000 images for 10 epochs; the experimental model on 1,600 images for 3 epochs, "
            "so this is not a like-for-like architecture comparison.")

    st.markdown("#### 2. Confusion matrices")
    _image("confusion_matrices.png")

    st.markdown("#### 3. Training and validation curves (experimental model)")
    _image("training_curves.png")
    h = hist["history"]
    st.dataframe(pd.DataFrame(h).set_index("epoch").style.format(
        {"train_loss": "{:.4f}", "val_loss": "{:.4f}", "train_acc": "{:.1%}",
         "val_acc": "{:.1%}", "seconds": "{:.1f}"}), width="stretch")
    mem = hist.get("peak_process_memory_mb")
    st.caption(f"Training time {hist['total_training_seconds']:.0f} s on CPU · peak memory "
               f"{mem:.0f} MB · batch {hist['batch_size']} · seed {hist['seed']} · checkpoint "
               f"from epoch {hist['best_epoch_by_val_acc']} (best validation accuracy).")
    st.warning("Training accuracy keeps rising while validation accuracy does not: the model "
               "starts to **overfit** the small training set after the first epoch.")

    st.markdown("#### 4. Example predictions (including incorrect ones)")
    _image("example_predictions.png")
    pred_path = RESULTS / "test_predictions.csv"
    if pred_path.is_file():
        df = pd.read_csv(pred_path)
        df["image"] = df["path"].str.split("/").str[-1]
        wrong = df[~(df["cnn_correct"] & df["mnv3_correct"])]
        with st.expander(f"All {len(wrong)} test images where at least one model was wrong"):
            st.dataframe(wrong[["image", "true_label", "cnn_pred", "cnn_confidence",
                                "mnv3_pred", "mnv3_confidence"]].reset_index(drop=True),
                         width="stretch")
    st.caption("Confidence = how far the score is from 0.5. High confidence on a wrong "
               "answer happens; confidence is not proof.")

    st.markdown("#### 5. Supplementary checks (read with care)")
    rob = cmp_.get("supplementary_robustness_200")
    if rob:
        rows = {}
        for key, name in (("cnn", "Original CNN"), ("mnv3", "Experimental")):
            for pert, label in (("gaussian_blur_0.6", "blur 0.6"), ("upscale256_down32", "resample")):
                rows[f"{name} - {label}"] = {
                    "Accuracy": f"{rob[key][pert]['accuracy']:.1%}",
                    "FAKE recall": f"{rob[key][pert]['fake_recall']:.1%}",
                    "AUC": f"{rob[key][pert]['auc']:.3f}"}
        st.markdown("**Robustness** - the same 200 images after mild blur or resampling:")
        st.dataframe(pd.DataFrame(rows).T, width="stretch")
    full = cmp_.get("supplementary_full_cifake_test_20000")
    if full:
        st.markdown("**Full CIFAKE test folder (20,000 images)** - " + full["note"])
        st.dataframe(pd.DataFrame(
            {"Original CNN": {k: f"{full['cnn'][k]:.3f}" for k in ("accuracy", "auc", "fake_f1")},
             "Experimental": {k: f"{full['mnv3'][k]:.3f}" for k in ("accuracy", "auc", "fake_f1")}}),
            width="stretch")

    st.markdown("#### 6. Data and limitations")
    if split:
        w = split["whole_dataset_checks"]
        st.write(f"Split: 800/100/100 images per class (train/val/test), seed {split['seed']}. "
                 f"Near-duplicate rule: {split['near_duplicate_rule']}. "
                 f"Test candidates rejected as duplicates of training images: "
                 f"{sum(split['rejected_candidates'].values())}. "
                 f"Whole dataset: {w['test_images_identical_to_a_train_image']} test images are "
                 f"identical to a train image.")
    st.markdown(
        "- Only CIFAKE (32×32, Stable Diffusion) images were used; **no external test set has "
        "been evaluated yet**, so performance on other images is unknown.\n"
        "- 200 test images give wide confidence intervals; small differences are not meaningful.\n"
        "- One run, one seed, no hyperparameter tuning.\n"
        "- This is not a universal AI-image detector.")
