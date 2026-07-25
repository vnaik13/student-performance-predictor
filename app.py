import streamlit as st
import pandas as pd
import joblib


# Load trained model pipeline
pipeline = joblib.load("student_performance_pipeline.pkl")


# Page configuration
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# Custom title section
st.title("🎓 AI Student Performance Predictor")

st.write(
    "An intelligent machine learning system that predicts "
    "student exam performance using academic, personal, "
    "and environmental factors."
)


# Model information cards
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Algorithm",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="R² Score",
        value="0.67"
    )

with col3:
    st.metric(
        label="Prediction Error",
        value="±1.08 marks"
    )


st.divider()


st.subheader("📝 Enter Student Information")


# Three-column layout

col1, col2, col3 = st.columns(3)


# Academic Details

with col1:

    st.markdown("## 📚 Academic Details")

    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0,
        max_value=24,
        value=5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=75
    )

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=0,
        max_value=100,
        value=70
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0,
        max_value=24,
        value=7
    )

    tutoring_sessions = st.number_input(
        "Tutoring Sessions",
        min_value=0,
        max_value=10,
        value=1
    )

    physical_activity = st.number_input(
        "Physical Activity",
        min_value=0,
        max_value=10,
        value=3
    )



# Student Details

with col2:

    st.markdown("## 👨‍🎓 Student Details")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    learning_disabilities = st.selectbox(
        "Learning Disabilities",
        ["Yes", "No"]
    )

    parental_education = st.selectbox(
        "Parental Education Level",
        [
            "High School",
            "College",
            "Postgraduate"
        ]
    )

    family_income = st.selectbox(
        "Family Income",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    internet_access = st.selectbox(
        "Internet Access",
        [
            "Yes",
            "No"
        ]
    )

    distance = st.selectbox(
        "Distance From Home",
        [
            "Near",
            "Moderate",
            "Far"
        ]
    )



# Environment Details

with col3:

    st.markdown("## 🏫 Learning Environment")

    parental_involvement = st.selectbox(
        "Parental Involvement",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    access_to_resources = st.selectbox(
        "Access To Resources",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    teacher_quality = st.selectbox(
        "Teacher Quality",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    school_type = st.selectbox(
        "School Type",
        [
            "Public",
            "Private"
        ]
    )

    peer_influence = st.selectbox(
        "Peer Influence",
        [
            "Positive",
            "Neutral",
            "Negative"
        ]
    )

    motivation = st.selectbox(
        "Motivation Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        [
            "Yes",
            "No"
        ]
    )



st.divider()


# Prediction button

if st.button(
    "🔮 Predict Exam Score",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Parental_Involvement": [parental_involvement],
        "Access_to_Resources": [access_to_resources],
        "Extracurricular_Activities": [extracurricular],
        "Sleep_Hours": [sleep_hours],
        "Previous_Scores": [previous_scores],
        "Motivation_Level": [motivation],
        "Internet_Access": [internet_access],
        "Tutoring_Sessions": [tutoring_sessions],
        "Family_Income": [family_income],
        "Teacher_Quality": [teacher_quality],
        "School_Type": [school_type],
        "Peer_Influence": [peer_influence],
        "Physical_Activity": [physical_activity],
        "Learning_Disabilities": [learning_disabilities],
        "Parental_Education_Level": [parental_education],
        "Distance_from_Home": [distance],
        "Gender": [gender]

    })


    prediction = pipeline.predict(input_data)[0]


    st.success(
        f"🎯 Predicted Exam Score: {prediction:.2f}"
    )


    if prediction >= 80:
        st.info(
            "🌟 Excellent academic performance expected!"
        )

    elif prediction >= 60:
        st.info(
            "👍 Good performance expected. "
            "Consistent improvement can increase results."
        )

    else:
        st.warning(
            "📌 Additional academic support may help improve performance."
        )


st.divider()

st.caption(
    "Built using Machine Learning | Random Forest Regression | Streamlit"
)