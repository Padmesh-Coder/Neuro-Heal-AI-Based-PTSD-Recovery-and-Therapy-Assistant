from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import os
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini
from langchain_groq import ChatGroq #Import ChatGroq
import speech_recognition as sr  # Import Speech Recognition
import tempfile  # Import tempfile
import wave
import contextlib

app = Flask(__name__)

# Load trained models
try:
    text_model = tf.keras.models.load_model("models/dass21_text_model.h5")
    encoder = joblib.load("models/encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Load audio emotion model
MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
try:
    audio_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, do_normalize=True)
    audio_id2label = audio_model.config.id2label
except Exception as e:
    print(f"Error loading audio model: {e}")
    audio_model = None  # Disable audio functionality if loading fails
    audio_feature_extractor = None
    audio_id2label = None

# Label mapping
label_mapping = {1: "Normal", 2: "Mild", 3: "Moderate", 4: "Severe", 5: "Extremely Severe"}

# Initialize ChatGroq (replace with your actual API key)
try:
    groq_llm = ChatGroq(
        temperature=0.3,
        groq_api_key="Your_Groq_API_Key",
        model_name="llama-3.3-70b-versatile"  # Model for summarization - Use correct model name
    )
except Exception as e:
    print(f"Error initializing ChatGroq: {e}")
    groq_llm = None  # Disable LLM functionality if initialization fails

# Initialize Gemini API
API_KEY = "Your_Gemini_API_Key"  # Replace with your actual API key - VERY IMPORTANT
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", api_key=API_KEY, temperature=0.7, top_p=0.9)

# 🌿 PTSD Recovery Chatbot Prompt
ptsd_prompt = """
You are a **kind, empathetic, and supportive AI therapist** specializing in **PTSD recovery** through **CBT (Cognitive Behavioral Therapy)**.

💙 **Your Role:**
- Gently assess PTSD symptoms step-by-step.
- Detect **severe distress or crisis situations** and respond appropriately.
- Offer **personalized CBT-based coping techniques**.
- Provide **medical advice in a non-intimidating, supportive way**.
- If the user expresses suicidal thoughts, **immediately provide crisis helplines & emergency support**.

🔍 **How You Diagnose PTSD:**
1. Start with a simple **emotional check-in**.
2. Gradually assess PTSD symptoms in a natural conversation flow.
3. If severe distress is detected, **immediately provide crisis support**.
4. Provide **recovery recommendations** (CBT exercises, mindfulness, breathing techniques).
5. Encourage **professional medical help when necessary**.

🩵 **Example Style:**
User: I feel anxious all the time...
Chatbot: I hear you. Anxiety can be overwhelming, but you're not alone. Let’s try a quick grounding exercise:
**Take a deep breath in… hold for 4 seconds… and slowly exhale.**
Would you like me to suggest a simple CBT thought-reframing technique? 💙

User: I can't sleep at night because of nightmares.
Chatbot: That sounds tough. PTSD-related nightmares can feel very real, but there are ways to manage them.
One effective technique is **imagery rehearsal therapy (IRT)**—reimagining the ending of a nightmare to feel safer.
Would you like some relaxation exercises before bed? 🌙✨

Now, let’s continue. How are you feeling today?
User: {user_input}
Chatbot:
"""

# Add conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")

# Create PTSD chatbot chain with memory
prompt = PromptTemplate(template=ptsd_prompt, input_variables=["user_input"])
chatbot_chain = LLMChain(prompt=prompt, llm=gemini_llm, memory=memory) # Gemini LLM

# 🔹 Function to assess PTSD level based on user input
def assess_ptsd(user_input):
    user_input = user_input.lower()

    crisis_keywords = ["suicidal", "end my life", "don't want to live", "kill myself", "give up"]
    high_risk_keywords = ["nightmares", "can't sleep", "flashbacks", "panic attacks", "intrusive thoughts"]
    moderate_risk_keywords = ["anxious", "nervous", "stressed", "on edge", "overwhelmed"]
    mild_keywords = ["fine", "okay", "good", "better now", "feeling alright"]

    if any(keyword in user_input for keyword in crisis_keywords):
        return "Severe (Immediate Crisis)"
    elif any(keyword in user_input for keyword in high_risk_keywords):
        return "Moderate to High"
    elif any(keyword in user_input for keyword in moderate_risk_keywords):
        return "Mild to Moderate"
    elif any(keyword in user_input for keyword in mild_keywords):
        return "Mild"
    else:
        return "Uncertain (Needs more information)"


# 🔹 Function to provide recommendations based on PTSD level
def get_recommendations(severity):
    recommendations = {
        "Severe (Immediate Crisis)": "⚠️ It sounds like you're in distress. Please reach out to a crisis helpline or a trusted person immediately. You are not alone! 💙",
        "Moderate to High": "I recommend grounding exercises, deep breathing, and CBT techniques like imagery rehearsal for nightmares. Would you like some relaxation exercises? 🌙✨",
        "Mild to Moderate": "It sounds like you're experiencing some anxiety or stress. Try journaling, mindfulness, or a guided breathing exercise. Let me know if you’d like to try one! 🧘‍♂️💙",
        "Mild": "That’s good to hear! Even if you're feeling okay, self-care is important. Do you want tips on maintaining emotional well-being? 😊",
        "Uncertain (Needs more information)": "I need a little more information to assess your situation. Would you like to share how you've been feeling lately? 💙"
    }
    return recommendations.get(severity, "Would you like to talk more about what’s on your mind? 💙")


# 🔹 Function to check if PTSD assessment should be displayed
def should_display_assessment(user_input):
    """Skip PTSD assessment if user expresses gratitude, closure, or positivity."""
    positive_closure_keywords = [
        "thank you", "thanks", "i feel better", "better now", "appreciate it",
        "i'm okay now", "i feel fine", "no worries", "i'm good", "feeling alright"
    ]
    return not any(keyword in user_input.lower() for keyword in positive_closure_keywords)

def process_chatbot_message(user_query):
    """Central function to process a chatbot message using Gemini."""
    # Step 1: Assess PTSD severity
    severity = assess_ptsd(user_query)

    # Step 2: Get chatbot response
    response = chatbot_chain.invoke({"user_input": user_query})

    # Step 3: Get PTSD-specific recommendations
    recommendation = get_recommendations(severity)

    # Step 4: Decide if PTSD assessment should be shown
    if should_display_assessment(user_query):
        final_response = f"{response['text']}\n\n🩵 **PTSD Assessment:** {severity}\n🌿 **Recommendation:** {recommendation}"
    else:
        final_response = response["text"]  # No PTSD assessment, just chatbot reply

    return final_response
# OLD LLM Logic - We don't need the functions anymore
def generate_llm_response(user_message):
    """Generates a response using Gemini"""
    final_response = process_chatbot_message(user_message)
    return final_response

def generate_initial_llm_response(severity, predicted_emotion=None, user_input=None):
    """Generates a CBT-based response using Langchain ChatGroq with Llama 3."""

    if severity == "Normal":
        prompt = f"""Based on CBT principles, generate a reassuring message for someone feeling normal. Acknowledge their well-being and then suggest they practice gratitude journaling, explaining briefly how it can improve their positive outlook. User Input: {user_input if user_input else 'None'}"""
    elif severity == "Mild":
        prompt = f"""Based on CBT principles, generate a suggestion for someone suffering from mild PTSD. Choose ONE: either suggest a relaxation exercise (describe how to do it) OR suggest a specific breathing technique (describe how to do it) to manage anxiety or stress. User Input: {user_input if user_input else 'None'}"""
    elif severity == "Moderate":
        prompt = f"""Based on CBT principles, generate a recommendation for someone suffering from moderate PTSD. Describe EITHER a specific journaling prompt they could use to explore their thoughts, OR a brief explanation of what exposure therapy is and how it can help them (with professional guidance). User Input: {user_input if user_input else 'None'}"""
    elif severity == "Severe":
        prompt = f"""Based on CBT principles, provide guidance for someone suffering from severe PTSD. Suggest professional support and then choose ONE of the following grounding techniques and describe it in detail, so the user can perform it:
        - 5-4-3-2-1 Exercise
        - Deep Breathing
        - Progressive Muscle Relaxation.
        Also include hotline information.
        User Input: {user_input if user_input else 'None'}"""
    else:
        prompt = f"""Based on CBT principles, generate a response for someone suffering from extremely severe PTSD. Emphasize immediate safety and seeking professional help immediately. Choose ONE immediate coping mechanism or grounding technique and describe it in detail. Provide emergency contact information and encourage them to reach out for crisis support. User Input: {user_input if user_input else 'None'}"""

    if predicted_emotion:
        prompt += f" Also consider that the user is feeling {predicted_emotion}."

    try:
        if groq_llm:  # Check if groq_llm is initialized
            llm_response = groq_llm.invoke(prompt).content
        else:
            return "LLM is unavailable. Please check the API key and model loading."
    except Exception as e:
        return f"Error generating response: {e}"

    return llm_response

# Define DASS-21 question mapping to survey feature names
survey_columns = [
    "Q3_1_S1", "Q3_2_S2", "Q3_3_S3", "Q3_4_S4", "Q3_5_S5", "Q3_6_S6", "Q3_7_S7",
    "Q3_8_A1", "Q3_9_A2", "Q3_10_A3", "Q3_11_A4", "Q3_12_A5", "Q3_13_A6", "Q3_14_A7",
    "Q3_15_D1", "Q3_16_D2", "Q3_17_D3", "Q3_18_D4", "Q3_19_D5", "Q3_20_D6", "Q3_21_D7"
]

# Define DASS-21 questions
dass21_questions = [
    "I found it hard to wind down", "I was aware of dryness of my mouth",
    "I couldn’t seem to experience any positive feeling at all", "I experienced breathing difficulty",
    "I found it difficult to work up the initiative to do things", "I tended to over-react to situations",
    "I experienced trembling", "I felt that I was using a lot of nervous energy",
    "I was worried about situations in which I might panic and make a fool of myself", "I felt that I had nothing to look forward to", "I found myself getting agitated",
    "I found it difficult to relax", "I felt down-hearted and blue",
    "I was intolerant of anything that kept me from getting on with what I was doing",
    "I felt I was close to panic", "I was unable to become enthusiastic about anything",
    "I felt I wasn’t worth much as a person", "I felt that I was rather touchy",
    "I was aware of the action of my heart in the absence of physical exertion",
    "I felt scared without any good reason", "I felt that life was meaningless"
]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", dass21_questions=dass21_questions)


@app.route("/process", methods=["POST"])
def process():
    try:
        # Collect demographic inputs
        demographic_inputs = {
            "Age": int(request.form["age"]),
            "Gender": request.form["gender"],
            "Marital Status": request.form["marital_status"],
            "Education": request.form["education"],
            "Occupation": request.form["occupation"],
            "Sleeping Problems": request.form["sleeping_problems"]
        }

        # Encode categorical inputs
        demographic_inputs["Gender"] = {"Male": 0, "Female": 1, "Other": 2}[demographic_inputs["Gender"]]
        demographic_inputs["Marital Status"] = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}[
            demographic_inputs["Marital Status"]]
        demographic_inputs["Education"] = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}[
            demographic_inputs["Education"]]
        demographic_inputs["Sleeping Problems"] = {"No": 0, "Yes": 1}[demographic_inputs["Sleeping Problems"]]

        # Collect survey responses
        survey_responses = [int(request.form[f"q{i}"]) for i in range(1, 22)]

        # Convert user responses into DataFrame
        df_survey = pd.DataFrame([survey_responses], columns=survey_columns)

        # Merge demographic data with survey responses
        df_demo = pd.DataFrame([demographic_inputs.values()], columns=encoder.feature_names_in_)
        df_input = pd.concat([df_demo, df_survey], axis=1)

        # Preprocess inputs
        X_demo = encoder.transform(df_input[encoder.feature_names_in_])
        X_survey = scaler.transform(df_input[survey_columns])
        X_text = np.hstack((X_demo, X_survey))

        # Predict using the text model
        text_prediction = text_model.predict(X_text)
        text_label = np.argmax(text_prediction[0]) + 1
        predicted_severity = label_mapping[text_label]

        confidence_adjustment = ""
        predicted_emotion = None

        # Handle audio file (if provided)
        audio_file = request.files["audio_file"]
        if audio_file and audio_model and audio_feature_extractor and audio_id2label: #Check audio model and feature extractor are loaded succesfully:
            try:
                audio_path = "temp_audio_questionnaire.wav"  # Unique filename for questionnaire audio
                audio_file.save(audio_path)

                predicted_emotion = predict_emotion(audio_path, audio_model, audio_feature_extractor, audio_id2label)

                # Adjust severity confidence based on emotion
                negative_emotions = ["sadness", "fear", "anger"]
                if predicted_emotion.lower() in negative_emotions:
                    confidence_adjustment = "(High Confidence)"
            except Exception as e:
                print(f"Audio processing error: {e}")
                predicted_emotion = "Error processing audio"
        else:
            predicted_emotion = None

        # Generate LLM response for results page
        llm_response = generate_initial_llm_response(predicted_severity, predicted_emotion) # Groq LLM


        return render_template("results.html",
                               predicted_severity=predicted_severity,
                               confidence_adjustment=confidence_adjustment,
                               predicted_emotion=predicted_emotion,
                               llm_response=llm_response)  # Pass llm_response

    except Exception as e:
        return render_template("index.html", error=str(e), dass21_questions=dass21_questions)  # Go back to questions page with error


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_message = request.form["message"]
        final_response = process_chatbot_message(user_message)  # Use the Gemini logic for Chatbot page
        return jsonify({"response": final_response})
    else:
        return render_template("chatbot.html")

def check_wav_format(file_path):
        try:
            with contextlib.closing(wave.open(file_path,'r')) as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                num_frames = wf.getnframes()
                comp_type = wf.getcomptype()
                comp_name = wf.getcompname()
                print(f"Channels: {num_channels}, Sample Width: {sample_width}, Frame Rate: {frame_rate}, Frames: {num_frames}, Compression Type: {comp_type}, Compression Name: {comp_name}")

                # Perform additional checks here, e.g., ensure sample width is 2 bytes (16-bit), etc.
                if sample_width != 2: #Checking if the sample is 2 bytes
                    return False, "Sample width is not 2 bytes(16-bit)"
                return True,"Valid WAV format"

        except wave.Error as e:
            return False, f"Invalid WAV file: {e}"
        except Exception as e:
            return False, f"Error checking WAV file: {e}"

@app.route("/process_audio", methods=["POST"])
def process_audio():
    try:
        r = sr.Recognizer()
        audio_file = request.files["audio"] #Get audio file

        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                audio_file.save(temp_audio.name) #Save to temporary directory

            #Checking WAV Format using check_wav_format Function
            is_valid_wav, message = check_wav_format(temp_audio.name)

            if not is_valid_wav:
                os.remove(temp_audio.name)
                return jsonify({"response": f"Invalid WAV file: {message}"})

            with sr.AudioFile(temp_audio.name) as source:
                audio = r.record(source)

            try:
                text = r.recognize_google(audio)
                final_response = process_chatbot_message(text) # Gemini LLM for Chatbot
                return jsonify({"response": final_response})
            except sr.UnknownValueError:
                return jsonify({"response": "Google Speech Recognition could not understand audio"})
            except sr.RequestError as e:
                return jsonify({"response": f"Could not request results from Google Speech Recognition service; {e}"})
            except Exception as e:
                return jsonify({"response": f"Speech Recognition Error: {str(e)}"})
            finally:
                os.remove(temp_audio.name) #Delete temporary file

        else:
            return jsonify({"response": "No audio file received."})
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"response": f"Error processing audio: {str(e)}"})

@app.route("/chatbot_audio", methods=["POST"])
def chatbot_audio():
    try:
        r = sr.Recognizer()
        audio_file = request.files["audio"]  # Get audio file

        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                audio_file.save(temp_audio.name)  # Save to temporary directory

            # Checking WAV Format using check_wav_format Function
            is_valid_wav, message = check_wav_format(temp_audio.name)

            if not is_valid_wav:
                os.remove(temp_audio.name)
                return jsonify({"response": f"Invalid WAV file: {message}"})

            with sr.AudioFile(temp_audio.name) as source:
                audio = r.record(source)

            try:
                text = r.recognize_google(audio)
                final_response = process_chatbot_message(text)  # Gemini LLM for Chatbot
                return jsonify({"response": final_response})
            except sr.UnknownValueError:
                return jsonify({"response": "Google Speech Recognition could not understand audio"})
            except sr.RequestError as e:
                return jsonify({"response": f"Could not request results from Google Speech Recognition service; {e}"})
            except Exception as e:
                return jsonify({"response": f"Speech Recognition Error: {str(e)}"})
            finally:
                os.remove(temp_audio.name)  # Delete temporary file

        else:
            return jsonify({"response": "No audio file received."})

    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"response": f"Error processing audio: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
