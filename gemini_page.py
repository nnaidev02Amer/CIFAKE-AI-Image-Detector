"""'Gemini AI Analysis' page: the original CNN (primary) next to a Gemini visual second opinion.

Reuses the image already analysed on the Image Detection page (no second uploader).
Gemini is called only when the user ticks the privacy acknowledgement and clicks the
button; the result is kept in the session for that image + model, so page reruns do
not make new requests. The API key value is never displayed.
"""
from __future__ import annotations

import hashlib

import streamlit as st

import gemini_analyzer as ga

LABELS = {"POSSIBLY_AI_GENERATED": ("Possibly AI-generated", "🟠"),
          "POSSIBLY_REAL": ("Possibly real", "🔵"),
          "INCONCLUSIVE": ("Inconclusive", "⚪")}
STATE_STYLE = {"AGREE": st.success, "DISAGREE": st.warning,
               "INCONCLUSIVE": st.info, "UNAVAILABLE": st.info}


def _secrets():
    """st.secrets raises if no secrets.toml exists; treat that as 'no secrets'."""
    try:
        return st.secrets if "GEMINI_API_KEY" in st.secrets else None
    except Exception:  # noqa: BLE001
        return None


def render() -> None:
    st.subheader("🤖 Gemini AI Analysis - second opinion")
    st.caption("The original CNN remains the primary model. Gemini gives an independent, "
               "visual second opinion. It is **not** a validated AI-image detector, and its "
               "accuracy on this task has not been measured in this project.")

    res = st.session_state.get("last")
    if not res or "original_bytes" not in res:
        st.info("No image analysed yet. Go to **Image Detection**, upload an image (or pick a "
                "sample) and then come back to this page.")
        return

    data: bytes = res["original_bytes"]
    pred = res["pred"]
    model = ga.get_model_name()
    api_key = ga.resolve_api_key(_secrets())
    cache_key = f"{hashlib.sha256(data).hexdigest()}:{model}"
    results = st.session_state.setdefault("gemini_results", {})
    gemini = results.get(cache_key)

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Uploaded image**")
        st.image(res["image"], width="stretch")
        w, h = res["image"].size
        st.caption(f"{st.session_state.get('last_name', 'image')} - {w}x{h}px")
    with right:
        st.markdown("##### 1. Original CNN (primary model)")
        c1, c2 = st.columns(2)
        c1.metric("CNN prediction", pred["label"])
        c2.metric("Score P(REAL)", f"{pred['p_real']:.4f}")
        st.caption(f"CNN confidence {pred['confidence']:.1%}. The CNN sees a 32x32 copy; "
                   "its score is not a probability of truth.")

        st.markdown("##### 2. Gemini visual assessment (second opinion)")
        if gemini is None:
            st.caption("Not requested yet for this image.")
        elif gemini.ok:
            text, icon = LABELS[gemini.assessment]
            st.markdown(f"**Assessment:** {icon} {text}")
            st.write(gemini.explanation)
        else:
            st.error(gemini.message)

    # ── Request section (user-triggered only) ──────────────────────────────────
    st.markdown("---")
    sdk_ok = ga.sdk_available()
    status = [f"Model: `{model}`",
              "API key: configured (value hidden)" if api_key else "API key: **not configured**",
              "SDK: installed" if sdk_ok else "SDK: **google-genai not installed**"]
    st.caption(" · ".join(status))
    if not api_key or not sdk_ok:
        st.info("Gemini is unavailable, so the app runs in **CNN-only** mode. See "
                "`docs/GEMINI_SETUP.md` to configure GEMINI_API_KEY locally.")

    st.warning(
        "**Privacy notice - read before sending.** Clicking the button sends this image to "
        "Google's Gemini API. A copy with all metadata removed (EXIF, including GPS location "
        "and camera details, XMP, comments) is sent at full resolution; your original file "
        "is not changed or stored by this app. On the Gemini API **free tier**, Google may "
        "use submitted content to improve its products and human reviewers may read it, so "
        "**do not send personal, sensitive or confidential images**. Quotas and pricing "
        "depend on your Google account and the selected model.")
    agreed = st.checkbox("I understand that this image will be sent to Google's Gemini API.",
                         key=f"gemini_ack_{cache_key}")
    clicked = st.button("Send to Gemini for a second opinion", type="primary",
                        disabled=not (agreed and api_key and sdk_ok))
    if clicked:
        with st.spinner(f"Asking {model}... (one request, no automatic retry)"):
            gemini = ga.analyze_image(data, api_key, model)
        results[cache_key] = gemini
        st.rerun()

    if gemini is not None:
        if gemini.ok:
            with st.expander("Gemini evidence and limitations", expanded=True):
                st.markdown("**Observable evidence (as reported by Gemini)**")
                if gemini.evidence:
                    st.markdown("\n".join(f"- {e}" for e in gemini.evidence))
                else:
                    st.caption("Gemini reported no specific visual evidence.")
                st.markdown("**Limitations stated by Gemini**")
                st.write(gemini.limitations)
                st.caption(f"Model {gemini.model} · {gemini.seconds:.1f} s · {gemini.image_note}")
        else:
            st.caption("You can try again later with the button above. The app never retries "
                       "automatically.")

    # ── Comparison (no combined score) ──────────────────────────────────────────
    st.markdown("##### 3. Comparison")
    cmp_ = ga.compare(pred["label"], gemini, pred["confidence"])
    STATE_STYLE[cmp_.state](f"**{cmp_.title}.** {cmp_.message}")
    for note in cmp_.notes:
        st.caption(note)
    st.info("Agreement between two systems does **not** establish that an image is authentic "
            "or AI-generated: both can be wrong in the same way. The CNN score and Gemini's "
            "text are shown separately and are never combined into one confidence number.")
