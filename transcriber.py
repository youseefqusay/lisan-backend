"""
transcriber.py
----------------
هذا الملف مسؤول عن:
1. تحميل نموذج Whisper (نسخة faster-whisper المجانية والسريعة)
2. تحويل أي ملف صوتي إلى نص
3. اكتشاف لغة المتحدث تلقائياً

النموذج يعمل بالكامل على جهازك (بدون إنترنت بعد أول تحميل) وبدون أي اشتراك.
"""

from faster_whisper import WhisperModel

# ---------------------------------------------------------
# إعدادات النموذج
# ---------------------------------------------------------
# نستخدم حجم "small" كتوازن جيد بين الدقة والسرعة على جهاز عادي.
# الأحجام المتاحة (من الأسرع/الأقل دقة إلى الأبطأ/الأدق):
# tiny -> base -> small -> medium -> large-v3
MODEL_SIZE = "small"

# "cpu" يشتغل على أي جهاز، "cuda" أسرع بكثير لو عندك كرت شاشة NVIDIA
DEVICE = "cpu"

# compute_type يتحكم بدقة الحسابات، int8 أخف وأسرع على المعالج العادي
COMPUTE_TYPE = "int8"

print(f"[transcriber] تحميل نموذج Whisper ({MODEL_SIZE})... أول مرة قد تأخذ وقت لتحميل النموذج.")
_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("[transcriber] تم تحميل النموذج بنجاح.")


def transcribe_audio(audio_path: str):
    """
    تحوّل ملف صوتي إلى نص وتكتشف اللغة تلقائياً.

    المدخلات:
        audio_path: مسار ملف الصوت (mp3, wav, m4a, ...)

    المخرجات: dict فيه
        {
            "language": "ar",              # رمز اللغة المكتشفة
            "language_probability": 0.98,  # نسبة الثقة بكشف اللغة
            "text": "النص الكامل هنا",
            "segments": [                  # النص مقسم لمقاطع مع التوقيت
                {"start": 0.0, "end": 3.2, "text": "..."},
                ...
            ]
        }
    """
    segments_iter, info = _model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,  # يكشف فترات السكوت ويتجاهلها قبل المعالجة —
        # يمنع مشكلة شائعة بـ Whisper اسمها "Hallucination" حيث يطلع
        # عبارات وهمية (مثل "Thank you for watching") بفترات السكوت،
        # لأن النموذج اتدرّب على كمية كبيرة من فيديوهات يوتيوب المترجمة
        # تلقائياً واللي أغلبها تنتهي بهذي العبارة بالذات.
        vad_parameters=dict(min_silence_duration_ms=1000, threshold=0.3),
    )

    segments = []
    full_text_parts = []

    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "text": " ".join(full_text_parts),
        "segments": segments,
    }


if __name__ == "__main__":
    # اختبار سريع: شغّل هذا الملف مباشرة مع مسار ملف صوتي للتجربة
    import sys
    if len(sys.argv) < 2:
        print("الاستخدام: python transcriber.py path/to/audio.mp3")
    else:
        result = transcribe_audio(sys.argv[1])
        print("اللغة المكتشفة:", result["language"], f"(ثقة: {result['language_probability']})")
        print("النص:", result["text"])
