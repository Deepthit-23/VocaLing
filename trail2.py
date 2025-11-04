import os
import uuid
import logging
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from transformers import pipeline
from gtts import gTTS
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask App Setup
app = Flask(__name__)
CORS(app)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback_secret')
Session(app)

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI Model (Replace with GroqCloud API later if needed)
conversation_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Speech Recognizer
recognizer = sr.Recognizer()

# Helper Functions
def record_audio(duration=5, sample_rate=44100):
    """Record audio from the microphone."""
    audio_path = f"static/recording_{uuid.uuid4()}.wav"
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        sf.write(audio_path, recording, sample_rate)
        return audio_path
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        return None

def assess_pronunciation(audio_path, target_text):
    """Assess the pronunciation quality."""
    try:
        # Load and transcribe the audio file
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            transcription = recognizer.recognize_google(audio)

        # Calculate similarity score
        similarity = SequenceMatcher(None, transcription.lower(), target_text.lower()).ratio()
        score = similarity * 100

        # Feedback based on score
        if score >= 90:
            feedback = "Excellent pronunciation! You sound almost native."
        elif score >= 75:
            feedback = "Good job! A few minor improvements could make your pronunciation even better."
        elif score >= 50:
            feedback = "Keep practicing. Focus on your intonation and stress."
        else:
            feedback = "Let's work on your pronunciation. Try listening to native speakers and mimicking their sounds."

        return {"score": score, "feedback": feedback, "transcription": transcription}
    except Exception as e:
        logger.error(f"Pronunciation assessment error: {e}")
        return {"error": str(e)}

def text_to_speech(response_text):
    """Convert text to speech and save as an audio file."""
    try:
        tts = gTTS(response_text)
        audio_path = f"static/response_{uuid.uuid4()}.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        return None

# Flask Routes
@app.route('/')
def home():
    """Home page."""
    session['conversation'] = [
        {"role": "system", "content": "You are a friendly language learning assistant called VocaLing."}
    ]
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle incoming questions and provide responses."""
    try:
        # Get user input
        user_input = request.json.get('text', '')

        # Get the conversation history
        conversation = session.get('conversation', [])
        conversation.append({"role": "user", "content": user_input})

        # Generate AI response
        response = conversation_pipeline(
            f"{user_input} | You are a language teacher. Be interactive, ask questions, take quizzes, and roleplay.",
            max_length=100
        )[0]['generated_text']
        conversation.append({"role": "assistant", "content": response})
        session['conversation'] = conversation

        # Convert AI response to speech
        audio_path = text_to_speech(response)

        return jsonify({
            "response": response,
            "audio_url": audio_path
        })
    except Exception as e:
        logger.error(f"Error in /ask route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/record', methods=['POST'])
def record():
    """Record user's audio and assess pronunciation."""
    try:
        # Record audio
        target_text = request.json.get('target_text', '')
        audio_path = record_audio()

        # Assess pronunciation
        assessment = assess_pronunciation(audio_path, target_text)

        return jsonify(assessment)
    except Exception as e:
        logger.error(f"Error in /record route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests."""
    return '', 204

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
