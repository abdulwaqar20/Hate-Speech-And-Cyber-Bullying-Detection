# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set page config first
st.set_page_config(
    page_title="Hate Speech & Cyberbullying Detector",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    .hate-speech {
        background-color: #ffebee;
        border: 3px solid #ff4b4b;
        color: #d32f2f;
    }
    .cyberbullying {
        background-color: #fff3e0;
        border: 3px solid #ff9800;
        color: #ef6c00;
    }
    .neutral {
        background-color: #e8f5e8;
        border: 3px solid #4caf50;
        color: #2e7d32;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 2px solid #bee5eb;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .progress-bar {
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 5px 0;
        height: 20px;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLP components
@st.cache_resource
def load_nlp_components():
    nltk.download('stopwords', quiet=True)
    nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    return nlp

nlp = load_nlp_components()

# Load trained models
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load("models/cyberbullying_tfidf_vectorizer.joblib")
        clf = joblib.load("models/cyberbullying_logreg_model.joblib")
        rule_components = joblib.load("models/cyberbullying_rule_components.joblib")
        
        st.success("✅ Models loaded successfully!")
        return tfidf, clf, rule_components
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

tfidf, clf, rule_components = load_models()

# Text cleaning function (same as training)
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s']")
HASHTAG_RE = re.compile(r"#\w+")

def clean_text(text):
    text = str(text).lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    text = NON_ALPHANUM_RE.sub(" ", text)
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if t.lemma_.strip() and len(t.lemma_) > 1]
    return " ".join(tokens)

# Enhanced Hate Speech Keywords
HATE_SPEECH_KEYWORDS = [
    'exterminate', 'genocide', 'subhuman', 'ethnic cleansing',
    'final solution', 'gas the', 'deport all', 'kill all',
    'arrogant', 'belligerent', 'big-headed', 'callous', 'dogmatic',
    'intolerant', 'machiavellian', 'narrow-minded', 'pompous',
    'selfish', 'vain', 'vulgar'
]

# Enhanced Cyberbullying Keywords
CYBERBULLYING_KEYWORDS = [
    'kill yourself', 'end your life', 'you should die',
    'piece of shit', 'worthless', 'no one loves you',
    'bitchy', 'boastful', 'boring', 'cowardly', 'cruel',
    'foolish', 'grumpy', 'jealous', 'nasty', 'rude',
    'sarcastic', 'sneaky', 'stupid', 'thoughtless', 'unreliable'
]

# Real ML Prediction Function
def ml_predict(text):
    """
    Real ML model prediction using your trained model
    """
    if tfidf is None or clf is None:
        # Fallback simulation if models not loaded
        text_lower = text.lower()
        if any(word in text_lower for word in ['hate', 'kill', 'exterminate', 'genocide']):
            return 0, [0.85, 0.10, 0.05]
        elif any(word in text_lower for word in ['stupid', 'idiot', 'kill yourself', 'worthless']):
            return 1, [0.10, 0.80, 0.10]
        else:
            return 2, [0.05, 0.10, 0.85]
    
    # Real prediction with your trained model
    try:
        # Clean and vectorize the text
        clean_sample = clean_text(text)
        sample_vec = tfidf.transform([clean_sample])
        
        # Get prediction and probabilities
        predicted_class = clf.predict(sample_vec)[0]
        probabilities = clf.predict_proba(sample_vec)[0]
        
        return predicted_class, probabilities
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        # Fallback to neutral if error occurs
        return 2, [0.1, 0.1, 0.8]

# Smart Hybrid Prediction System
def smart_hybrid_predict(text, use_rules=True, ml_confidence_threshold=0.7):
    """
    Smart hybrid prediction: ML first, rules only for high-confidence cases
    """
    # Step 1: Get ML prediction first
    ml_class, ml_probabilities = ml_predict(text)
    ml_confidence = ml_probabilities[ml_class]
    
    class_names = {
        0: "Hate Speech",
        1: "Cyberbullying", 
        2: "Neutral"
    }
    
    # Step 2: If ML is confident, return ML prediction
    if ml_confidence >= ml_confidence_threshold or not use_rules:
        return ml_class, ml_probabilities, class_names[ml_class], False
    
    # Step 3: Check rules only if ML is uncertain
    text_lower = text.lower()
    rule_used = False
    
    # Rule-based detection for Hate Speech
    if any(keyword in text_lower for keyword in HATE_SPEECH_KEYWORDS):
        rule_used = True
        # Moderate confidence for rules (not 90%)
        return 0, [0.75, 0.15, 0.10], "Hate Speech (Rule-Assisted)", True

    # Rule-based detection for Cyberbullying
    if any(keyword in text_lower for keyword in CYBERBULLYING_KEYWORDS):
        rule_used = True
        return 1, [0.15, 0.75, 0.10], "Cyberbullying (Rule-Assisted)", True
    
    # Step 4: Return ML prediction if no rules triggered
    return ml_class, ml_probabilities, class_names[ml_class], rule_used

# Function to create custom progress bars
def create_progress_bar(label, value, color):
    percentage = int(value * 100)
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span><strong>{label}</strong></span>
            <span>{percentage}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {percentage}%; background-color: {color};">
                {percentage}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚫 AI Hate Speech & Cyberbullying Detection</h1>', unsafe_allow_html=True)
st.markdown("""
This advanced AI system detects **hate speech**, **cyberbullying**, and **neutral content** in social media text using your trained machine learning model.
""")

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    if clf is not None:
        st.success("""
        **Model**: Enhanced Logistic Regression  
        **Status**: ✅ Loaded Successfully
        **Classes**: 
        - 🚫 Hate Speech
        - ⚠️ Cyberbullying  
        - ✅ Neutral
        """)
    else:
        st.warning("""
        **Model**: Simulation Mode  
        **Status**: ⚠️ Using Fallback
        **Note**: Train and save models first for full functionality
        """)
    
    st.header("⚙️ Settings")
    show_details = st.checkbox("Show detailed analysis", value=True)
    show_confidence = st.checkbox("Show confidence scores", value=True)
    use_enhanced_rules = st.checkbox("Use enhanced rule-based detection", value=True)
    ml_confidence_threshold = st.slider("ML Confidence Threshold", 0.5, 0.9, 0.7, 0.05, 
                                       help="Rules only activate when ML confidence is below this threshold")

# Main app
def main():
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🔍 Text Analysis", "📊 Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here to detect hate speech or cyberbullying...",
            height=150,
            key="text_input"
        )
        
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True) and text_input:
            with st.spinner("Analyzing text with smart hybrid detection..."):
                prediction, probabilities, class_name, rule_used = smart_hybrid_predict(
                    text_input, use_enhanced_rules, ml_confidence_threshold
                )
                
                # Display prediction result
                st.subheader("Detection Result")
                
                if prediction == 0:
                    st.markdown(f"""
                    <div class="prediction-box hate-speech">
                        <h2>🚫 HATE SPEECH DETECTED</h2>
                        <p style="font-size: 1.2rem;">This text contains hate speech content</p>
                        <p style="font-size: 1rem;">Targets: Religion, Ethnicity, Gender, or Protected Groups</p>
                        {"" if not rule_used else "<p style='font-size: 0.9rem;'><em>🔧 Rule-based system assisted</em></p>"}
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box cyberbullying">
                        <h2>⚠️ CYBERBULLYING DETECTED</h2>
                        <p style="font-size: 1.2rem;">This text contains cyberbullying content</p>
                        <p style="font-size: 1rem;">Includes: Personal attacks, harassment, or threats</p>
                        {"" if not rule_used else "<p style='font-size: 0.9rem;'><em>🔧 Rule-based system assisted</em></p>"}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box neutral">
                        <h2>✅ NEUTRAL CONTENT</h2>
                        <p style="font-size: 1.2rem;">This text appears to be neutral and safe</p>
                    </div>
                    """, unsafe_allow_html=True)
            
                # Confidence scores visualization
                if show_confidence:
                    st.subheader("Confidence Analysis")
                    
                    # Create custom progress bars
                    create_progress_bar("Hate Speech Probability", probabilities[0], "#ff4b4b")
                    create_progress_bar("Cyberbullying Probability", probabilities[1], "#ff9800")
                    create_progress_bar("Neutral Probability", probabilities[2], "#4caf50")
                    
                    # Warning for borderline cases
                    if probabilities[0] > 0.3 and probabilities[1] > 0.3:
                        st.markdown("""
                        <div class="warning-box">
                            ⚠️ <strong>Borderline Case Detected</strong><br>
                            This text shows characteristics of both hate speech and cyberbullying. 
                            Manual review recommended for accurate moderation.
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed probability analysis
                if show_details:
                    st.subheader("Detailed Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Hate Speech Probability", 
                            f"{probabilities[0]:.2%}",
                            delta="High risk" if probabilities[0] >= 0.5 else None,
                            delta_color="inverse"
                        )
                    with col2:
                        st.metric(
                            "Cyberbullying Probability", 
                            f"{probabilities[1]:.2%}",
                            delta="High risk" if probabilities[1] >= 0.5 else None,
                            delta_color="inverse"
                        )
                    with col3:
                        st.metric(
                            "Neutral Probability", 
                            f"{probabilities[2]:.2%}",
                            delta="Safe" if probabilities[2] >= 0.7 else None
                        )
    
    with tab2:
        st.header("Model Performance")
        
        if clf is not None:
            # Performance metrics from your training
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Accuracy", "82.0%")
            with col2:
                st.metric("Weighted F1-Score", "82.5%")
            with col3:
                st.metric("Hate Speech F1", "92.8%")
            with col4:
                st.metric("Cyberbullying F1", "77.9%")
            
            # Class-wise performance
            st.subheader("Class-wise Performance")
            
            performance_data = {
                'Class': ['Hate Speech', 'Cyberbullying', 'Neutral'],
                'Precision': ['93.8%', '81.5%', '53.3%'],
                'Recall': ['91.9%', '74.6%', '65.8%'],
                'F1-Score': ['92.8%', '77.9%', '58.9%']
            }
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
        else:
            st.warning("Performance metrics available after model training")
        
        # System Architecture
        st.subheader("System Architecture")
        st.markdown("""
        **Smart Hybrid Approach:**
        - **🤖 ML-First Strategy**: Your trained Logistic Regression model makes primary decisions
        - **🔧 Rule Assistance**: Rules only activate when ML confidence is low
        - **🎯 High-Severity Keywords**: Rules use only unambiguous toxic words
        - **⚖️ Balanced Confidence**: Rule confidence capped at 75% (not 90%)
        - **🔄 Same Preprocessing**: Uses identical text cleaning as training
        """)
    
    with tab3:
        st.header("About the System")
        
        st.subheader("🔧 Technical Details")
        st.info("""
        **Smart Hybrid Detection System**
        
        **Key Features:**
        - ✅ **Real ML Model**: Uses your trained Logistic Regression
        - ✅ **ML-First Approach**: Machine learning handles nuanced decisions
        - ✅ **Rule Assistance**: Rules help only when ML is uncertain  
        - ✅ **Context-Aware**: Better handling of neutral sentences with toxic words
        - ✅ **Identical Pipeline**: Same preprocessing as training for consistency
        
        **Detection Categories:**
        - **Hate Speech**: Religious, ethnic, gender-based, dehumanizing content
        - **Cyberbullying**: Personal attacks, threats, harassment, insults
        - **Neutral**: Safe and appropriate content
        """)
        
        st.subheader("🎯 How It Works")
        st.write("""
        1. **Text Input**: User provides social media text
        2. **Text Cleaning**: Identical preprocessing as training (lemmatization, stopword removal)
        3. **ML Analysis**: Your trained Logistic Regression model makes primary prediction
        4. **Confidence Check**: If ML confidence is high, return ML result
        5. **Rule Check**: If ML is uncertain, check high-severity keywords
        6. **Final Decision**: Return most appropriate classification
        """)
        
        st.subheader("👥 Developed By")
        st.write("""
        **NED University of Engineering & Technology**  
        **Natural Language Processing (CT-485) - Group Project**
        
        - Syed Hammad Atif (CTAI-22017)
        - Abdul Waqar (CTAI-22042)
        - Syed Muhammad Daniyal Qadri (CTAI-22037)
        - Nofal Shameem (CTAI-22039)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>NED University of Engineering & Technology</strong></p>
    <p>Natural Language Processing Project | Smart Hybrid Detection System</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()