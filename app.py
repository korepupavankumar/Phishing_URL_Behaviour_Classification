
import streamlit as st
import pandas as pd
import joblib
from urllib.parse import urlparse
import re
import ipaddress


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Phishing URL Behaviour Classification",
    page_icon="🔐",
    layout="centered"
)


# ==========================================
# Load Model
# ==========================================

MODEL_PATH = "model/url_phishing_model.pkl"
FEATURE_PATH = "model/url_feature_columns.pkl"

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
except Exception as e:
    st.error("Unable to load the trained model.")
    st.error(str(e))
    st.stop()


# ==========================================
# URL Feature Extraction
# ==========================================

def extract_url_features(url):

    # Add scheme if the user does not provide one
    original_url = url.strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", original_url):
        url = "http://" + original_url
    else:
        url = original_url

    parsed = urlparse(url)

    domain = parsed.netloc

    # Remove username/password if present
    if "@" in domain:
        domain = domain.split("@")[-1]

    # Remove port
    domain = domain.split(":")[0]

    # --------------------------------------
    # Basic URL values
    # --------------------------------------

    url_length = len(original_url)

    domain_length = len(domain)

    # --------------------------------------
    # Check whether domain is an IP address
    # --------------------------------------

    is_domain_ip = 0

    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    # --------------------------------------
    # Subdomain count
    # --------------------------------------

    domain_parts = domain.split(".")

    if len(domain_parts) > 2:
        no_of_subdomain = len(domain_parts) - 2
    else:
        no_of_subdomain = 0

    # --------------------------------------
    # Obfuscation features
    # --------------------------------------

    obfuscated_chars = url.count("%")

    no_of_obfuscated_char = obfuscated_chars

    has_obfuscation = 1 if obfuscated_chars > 0 else 0

    obfuscation_ratio = (
        obfuscated_chars / url_length
        if url_length > 0
        else 0
    )

    # --------------------------------------
    # Character statistics
    # --------------------------------------

    no_of_letters = sum(c.isalpha() for c in original_url)

    letter_ratio = (
        no_of_letters / url_length
        if url_length > 0
        else 0
    )

    no_of_digits = sum(c.isdigit() for c in original_url)

    digit_ratio = (
        no_of_digits / url_length
        if url_length > 0
        else 0
    )

    # --------------------------------------
    # Special URL characters
    # --------------------------------------

    no_of_equals = original_url.count("=")

    no_of_qmark = original_url.count("?")

    no_of_ampersand = original_url.count("&")

    special_characters = set(
        "!@#$%^&*()[]{};:'\"\\|,<>~`"
    )

    no_of_other_special_chars = sum(
        1 for c in original_url
        if c in special_characters
    )

    special_char_ratio = (
        no_of_other_special_chars / url_length
        if url_length > 0
        else 0
    )

    # --------------------------------------
    # HTTPS
    # --------------------------------------

    is_https = 1 if parsed.scheme.lower() == "https" else 0

    # --------------------------------------
    # Create feature dictionary
    # --------------------------------------

    features = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": no_of_obfuscated_char,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": no_of_letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": no_of_digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": no_of_other_special_chars,
        "SpacialCharRatioInURL": special_char_ratio,
        "IsHTTPS": is_https
    }

    return features


# ==========================================
# Header
# ==========================================

st.title("🔐 Phishing URL Behaviour Classification")

st.write(
    "Enter a website URL to classify its behaviour "
    "using a machine learning model."
)

st.info(
    "This prototype analyzes URL-based structural characteristics. "
    "It does not directly inspect the website's content."
)


# ==========================================
# URL Input
# ==========================================

url = st.text_input(
    "Enter Website URL",
    placeholder="https://www.example.com"
)


# ==========================================
# Prediction
# ==========================================

if st.button("🔍 Check URL", use_container_width=True):

    if not url.strip():

        st.warning("Please enter a URL.")

    else:

        try:

            # Extract features
            feature_data = extract_url_features(url)

            # Convert to DataFrame
            input_data = pd.DataFrame(
                [feature_data],
                columns=feature_columns
            )

            # Prediction
            prediction = model.predict(input_data)[0]

            # Probability
            probability = model.predict_proba(input_data)[0]

            phishing_probability = probability[1]
            legitimate_probability = probability[0]

            # ----------------------------------
            # Display result
            # ----------------------------------

            if prediction == 1:

                st.error("🚨 Potentially Phishing URL")

                st.write(
                    f"Model phishing probability: "
                    f"**{phishing_probability * 100:.2f}%**"
                )

            else:

                st.success("✅ Likely Legitimate URL")

                st.write(
                    f"Model legitimate probability: "
                    f"**{legitimate_probability * 100:.2f}%**"
                )

            # ----------------------------------
            # Feature information
            # ----------------------------------

            with st.expander("View extracted URL features"):

                feature_df = pd.DataFrame(
                    [feature_data]
                ).T

                feature_df.columns = ["Value"]

                st.dataframe(
                    feature_df,
                    use_container_width=True
                )

        except Exception as e:

            st.error("Unable to analyze this URL.")

            st.exception(e)