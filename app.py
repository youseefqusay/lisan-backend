"""
app.py
----------------
السيرفر الرئيسي (Flask). يستقبل الصوت من الواجهة، يمرره على:
  1. transcriber.py  -> تحويل الكلام إلى نص + كشف اللغة
  2. diarizer.py      -> تحديد مين قال شنو ومتى (تمييز المتحدثين)
  3. translator.py   -> ترجمة النص إلى اللغة المطلوبة
ثم يرجع النتيجة كـ JSON للواجهة الأمامية.
"""

import os
import uuid
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

from transcriber import transcribe_audio
from diarizer import diarize_audio, assign_speakers_to_transcript
from translator import translate_text, SUPPORTED_LANGUAGES
from summarizer import summarize_meeting

app = Flask(__name__)
CORS(app)  # يسمح للواجهة الأمامية (على منفذ مختلف) بالتواصل مع السيرفر


@app.after_request
def add_private_network_header(response):
    """
    Chrome له ميزة أمان اسمها Private Network Access تمنع صفحات/إضافات
    من الوصول لـ localhost تلقائياً. هذا الهيدر يصرّح لها بالوصول صراحة —
    بدونه، طلبات إضافة الكروم تفشل بصمت حتى لو السيرفر شغّال تمام.
    """
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def convert_to_wav(input_path: str) -> str:
    """
    يحوّل أي صيغة صوتية واصلة (webm, mp3, m4a...) إلى WAV قياسي
    (16kHz, قناة واحدة) باستخدام FFmpeg، عشان نضمن توافقها مع كل
    المكتبات المستخدمة (Whisper و pyannote) بدون مفاجآت.

    يتطلب تثبيت FFmpeg على الجهاز مسبقاً (راجع README.md لخطوات التثبيت).
    """
    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # يستبدل الملف لو موجود من قبل
                "-i", input_path,
                "-ar", "16000",  # معدل أخذ العينات المطلوب لـ Whisper
                "-ac", "1",      # قناة واحدة (mono)
                output_path,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg غير مثبّت على جهازك. ثبّته أولاً (راجع قسم "
            "\"تثبيت FFmpeg\" في README.md) ثم أعد تشغيل السيرفر."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"فشل تحويل الصوت بواسطة FFmpeg: {e.stderr.decode(errors='ignore')}")

    return output_path


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """يرجع قائمة اللغات المدعومة لعرضها بالواجهة الأمامية."""
    return jsonify(SUPPORTED_LANGUAGES)


@app.route("/api/transcribe", methods=["POST"])
def transcribe_endpoint():
    """
    يستقبل ملف صوتي + لغة الترجمة المطلوبة (اختياري)، ويرجع:
    - النص الأصلي
    - اللغة المكتشفة
    - النص المترجم (إذا طُلبت لغة ترجمة)
    """
    if "audio" not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف صوتي (audio)"}), 400

    audio_file = request.files["audio"]
    target_lang = request.form.get("target_lang")  # مثلاً "en"
    # الالتقاط المباشر (الإضافة) يرسل "false" هنا لأن تمييز المتحدثين
    # لا يعطي نتيجة موثوقة على مقاطع منفصلة قصيرة (راجع README لتفاصيل هذا القرار)
    should_diarize = request.form.get("diarize", "true") != "false"

    # حفظ الملف الخام مؤقتاً بصيغته الأصلية
    temp_filename = f"{uuid.uuid4().hex}_{audio_file.filename}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    audio_file.save(temp_path)

    wav_path = None
    try:
        # تحويل لصيغة WAV قياسية أولاً (يحل مشاكل توافق صيغة webm)
        wav_path = convert_to_wav(temp_path)

        # الخطوة 1: تحويل الصوت لنص + كشف اللغة
        result = transcribe_audio(wav_path)

        # الخطوة 2: تمييز المتحدثين (فقط للتسجيلات الكاملة، مو للمقاطع الحية القصيرة)
        if should_diarize:
            diarization_segments = diarize_audio(wav_path)
            output_segments = assign_speakers_to_transcript(
                result["segments"], diarization_segments
            )
        else:
            output_segments = result["segments"]

        response = {
            "detected_language": result["language"],
            "language_probability": result["language_probability"],
            "original_text": result["text"],
            "segments": output_segments,
        }

        # الخطوة 3: الترجمة (فقط إذا المستخدم طلب لغة هدف مختلفة)
        if target_lang and target_lang != result["language"]:
            translated = translate_text(
                result["text"],
                source_lang_whisper=result["language"],
                target_lang_whisper=target_lang,
            )
            response["translated_text"] = translated
            response["target_language"] = target_lang

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # حذف الملفات المؤقتة (الخام والمحوّل) بعد المعالجة
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


@app.route("/api/summarize", methods=["POST"])
def summarize_endpoint():
    """
    يستقبل نص اجتماع كامل (JSON) ويرجع ملخص + قائمة مهام/قرارات.

    body متوقع:
        {
            "text": "النص الكامل للاجتماع...",
            "source_lang": "ar",       // لغة النص (رمز Whisper)
            "output_lang": "ar"        // اختياري، لغة النتيجة (نفس المصدر افتراضياً)
        }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    source_lang = data.get("source_lang")
    output_lang = data.get("output_lang")  # اختياري

    if not text:
        return jsonify({"error": "لم يتم إرسال نص (text) للتلخيص"}), 400
    if not source_lang:
        return jsonify({"error": "لم يتم تحديد لغة النص (source_lang)"}), 400

    try:
        result = summarize_meeting(text, source_lang_whisper=source_lang, output_lang_whisper=output_lang)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # threaded=False: على جهاز بموارد محدودة، معالجة أكثر من طلب بالتوازي
    # تخلي الطلبات تتزاحم على نفس المعالج وتصير أبطأ من معالجتهم وحد ورا الثاني.
    # مقطع واحد وقتاً يشتغل بشكل مستقر أكثر من عدة مقاطع متزاحمة.
    # debug=True مفيد أثناء التطوير فقط، أطفئه عند التسليم النهائي
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
