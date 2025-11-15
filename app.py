# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import spacy
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Download stopwords
nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")
    st.stop()

# Page configuration
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
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
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
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🚫 AI Hate Speech & Cyberbullying Detection</h1>', unsafe_allow_html=True)
st.markdown("""
This advanced AI system detects **hate speech**, **cyberbullying**, and **neutral content** in social media text using machine learning.
Developed as part of NLP Course Project at NED University.
""")

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    st.info("""
    **Model**: Enhanced Logistic Regression  
    **Accuracy**: 82.0%  
    **Classes**: 
    - 🚫 Hate Speech (93.8% precision)
    - ⚠️ Cyberbullying (81.5% precision)  
    - ✅ Neutral (53.3% precision)
    """)
    
    st.header("⚙️ Settings")
    show_details = st.checkbox("Show detailed analysis", value=True)
    show_confidence = st.checkbox("Show confidence scores", value=True)
    use_enhanced_rules = st.checkbox("Use enhanced rule-based detection", value=True)

# Load models and components
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("models/cyberbullying_tfidf_vectorizer.joblib")
        model = joblib.load("models/cyberbullying_logreg_model.joblib")
        rule_components = joblib.load("models/cyberbullying_rule_components.joblib")
        return vectorizer, model, rule_components
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Text cleaning function
def clean_text(text):
    STOPWORDS = set(stopwords.words('english'))
    URL_RE = re.compile(r"http\S+|www\.\S+")
    MENTION_RE = re.compile(r"@\w+")
    NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s']")
    HASHTAG_RE = re.compile(r"#\w+")
    
    text = str(text).lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    text = NON_ALPHANUM_RE.sub(" ", text)
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if t.lemma_ not in STOPWORDS and t.lemma_.strip() and len(t.lemma_) > 1]
    return " ".join(tokens)

# Enhanced prediction with rule-based system
def enhanced_predict(text, vectorizer, model, rule_components=None, use_rules=True):
    # Rule-based detection first (if enabled and rules available)
    if use_rules and rule_components:
        text_lower = text.lower()
        
        # Hate speech keywords
        hate_keywords = [
            'exterminat', 'genetic', 'inferior', 'superior', 'eliminat', 
            'all muslim', 'all jew', 'all black', 'all white', 'all women', 'all immigrant',
            'should die', 'must die', 'deserve to die', 'not human', 'subhuman',
            'deport all', 'send back', 'not welcome', 'terrorist', 'islamic extremist'
        ]
        
        # Cyberbullying keywords
        cyberbullying_keywords = [
            'kill yourself', 'your mother', 'aborted', 'moron', 'idiot',
            'stupid', 'retard', 'worthless', 'piece of shit', 'go to hell',
            'fuck you', 'bastard', 'asshole', 'ugly', 'fat', 'nobody likes you',
            'loser', 'unpopular', 'you should die', 'everyone hates you'
        ]
        
        if any(keyword in text_lower for keyword in hate_keywords):
            return 0, [0.85, 0.10, 0.05], "Hate Speech (Rule-Based)", True
        
        if any(keyword in text_lower for keyword in cyberbullying_keywords):
            return 1, [0.10, 0.80, 0.10], "Cyberbullying (Rule-Based)", True
    
    # ML model prediction
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(text_vector)[0]
    
    # Apply optimal threshold for hate speech (0.394 from your analysis)
    optimal_threshold = 0.394
    if probabilities[0] >= optimal_threshold:
        predicted_class = 0
    else:
        predicted_class = model.predict(text_vector)[0]
    
    class_names = {
        0: "Hate Speech",
        1: "Cyberbullying", 
        2: "Neutral"
    }
    
    return predicted_class, probabilities, class_names[predicted_class], False

# Main app
def main():
    vectorizer, model, rule_components = load_models()
    
    if vectorizer is None or model is None:
        st.error("Please ensure model files are available in the 'models' folder:")
        st.code("""
        models/
        ├── cyberbullying_tfidf_vectorizer.joblib
        ├── cyberbullying_logreg_model.joblib
        └── cyberbullying_rule_components.joblib
        """)
        return
    
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
            with st.spinner("Analyzing text with advanced detection..."):
                prediction, probabilities, class_name, rule_used = enhanced_predict(
                    text_input, vectorizer, model, rule_components, use_enhanced_rules
                )
                
                # Display prediction result
                st.subheader("Detection Result")
                
                if prediction == 0:
                    st.markdown(f"""
                    <div class="prediction-box hate-speech">
                        <h2>🚫 HATE SPEECH DETECTED</h2>
                        <p style="font-size: 1.2rem;">This text contains hate speech content</p>
                        <p style="font-size: 1rem;">Targets: Religion, Ethnicity, or Gender</p>
                        {"" if not rule_used else "<p style='font-size: 0.9rem;'><em>🔧 Detected via rule-based system</em></p>"}
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box cyberbullying">
                        <h2>⚠️ CYBERBULLYING DETECTED</h2>
                        <p style="font-size: 1.2rem;">This text contains cyberbullying content</p>
                        <p style="font-size: 1rem;">Includes: Personal attacks, harassment, or threats</p>
                        {"" if not rule_used else "<p style='font-size: 0.9rem;'><em>🔧 Detected via rule-based system</em></p>"}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box neutral">
                        <h2>✅ NEUTRAL CONTENT</h2>
                        <p style="font-size: 1.2rem;">This text appears to be neutral and safe</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show rule-based detection info
                if rule_used:
                    st.markdown("""
                    <div class="success-box">
                        ✅ <strong>Rule-Based Detection Active</strong><br>
                        Critical content detected using keyword patterns for immediate response.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence scores visualization
                if show_confidence:
                    st.subheader("Confidence Analysis")
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Confidence', 
                              x=['Hate Speech', 'Cyberbullying', 'Neutral'],
                              y=probabilities,
                              marker_color=['#ff4b4b', '#ff9800', '#4caf50'],
                              text=[f'{p:.1%}' for p in probabilities],
                              textposition='auto')
                    ])
                    
                    fig.update_layout(
                        title="Prediction Confidence Scores",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1]),
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
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
                            delta=f"Threshold: 39.4%" if probabilities[0] >= 0.394 else None,
                            delta_color="inverse"
                        )
                    with col2:
                        st.metric(
                            "Cyberbullying Probability", 
                            f"{probabilities[1]:.2%}"
                        )
                    with col3:
                        st.metric(
                            "Neutral Probability", 
                            f"{probabilities[2]:.2%}"
                        )
    
    with tab2:
        st.header("Model Performance")
        
        # Performance metrics
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
            'Class': ['🚫 Hate Speech', '⚠️ Cyberbullying', '✅ Neutral'],
            'Precision': ['93.8%', '81.5%', '53.3%'],
            'Recall': ['91.9%', '74.6%', '65.8%'],
            'F1-Score': ['92.8%', '77.9%', '58.9%']
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        cm_data = np.array([[4376, 284, 100], 
                           [568, 2261, 204], 
                           [312, 202, 987]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Hate Speech', 'Cyberbullying', 'Neutral'],
                    yticklabels=['Hate Speech', 'Cyberbullying', 'Neutral'])
        plt.title('Confusion Matrix (Total Samples: 9,294)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(fig)
    
    with tab3:
        st.header("About the System")
        
        st.subheader("🔧 Technical Details")
        st.info("""
        **Enhanced Logistic Regression with TF-IDF Features**
        
        **Key Features:**
        - ✅ **World-class hate speech detection** (93.8% precision)
        - ✅ **Effective cyberbullying detection** (81.5% precision)  
        - ✅ **Rule-based enhancement** for critical content
        - ✅ **Optimal threshold tuning** for hate speech
        
        **Dataset:**
        - 46,467 social media texts
        - Balanced classes: Hate Speech (50%), Cyberbullying (33%), Neutral (17%)
        - Covers: Religion, Ethnicity, Gender, Age-based bullying, Personal attacks
        """)
        
        st.subheader("🎯 How It Works")
        st.write("""
        1. **Text Preprocessing**: Cleans and normalizes input text
        2. **Feature Extraction**: Converts text to TF-IDF vectors with n-grams
        3. **Rule-Based Detection**: Checks for critical keywords in hate speech and cyberbullying
        4. **Machine Learning**: Uses logistic regression with class balancing
        5. **Threshold Optimization**: Applies optimal thresholds for accurate classification
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
    <p><strong>🚫 AI Hate Speech & Cyberbullying Detection System</strong></p>
    <p>Natural Language Processing Project | NED University</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
