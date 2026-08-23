
"""
kb_recommender.py
─────────────────
Tra cứu knowledge base và gọi Groq LLM sinh khuyến nghị
dựa trên kết quả CNN phân loại bệnh lá sầu riêng.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
from pipeline.weather_kb import format_weather_summary_vi

# Force UTF-8 encoding for Windows console to support Vietnamese characters print
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

load_dotenv()



# ============================================================================
# 1. MAPPING
# ============================================================================

DISEASE_IDX_TO_LABEL: dict[int, str] = {
    0: "healthy",
    1: "algal",
    2: "allocaridara_attack",
    3: "blight",
    4: "phomopsis",
}

SEVERITY_IDX_TO_LEVEL: dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}


# ============================================================================
# 2. KNOWLEDGE BASE
# ============================================================================

class DurianLeafKB:
    def __init__(self, kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cases = data["cases"]
        self.sources = {s["source_id"]: s for s in data["sources"]}
        self.weather_scenarios = data["metadata"]["weather_scenarios"]
        self.safety_note = data["metadata"]["safety_note"]

    def lookup(self, disease_label, severity_level, weather_scenario):
        if disease_label == "healthy":
            severity_level = 0

        for case in self.cases:
            key = case["lookup_key"]
            if (
                key["disease_label"] == disease_label
                and key["severity_level"] == severity_level
                and key["weather_scenario"] == weather_scenario
            ):
                return case
        return None


    def get_source_titles(self, source_ids: list[str]) -> list[str]:

        results = []

        for sid in source_ids:

            src = self.sources.get(sid)

            if src:
                results.append(
                    f"[{sid}] {src['title']} — {src['url']}"
                )

        return results

def _check_condition(condition, weather):
    op_map = {
        ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b,
        "==": lambda a, b: a == b,
    }
    op = op_map[condition["operator"]]
    value = condition["value"]
    ctype = condition["type"]

    if ctype == "derived_field_threshold":
        return op(weather[condition["field"]], value)

    if ctype == "current_value":
        return op(weather[condition["field"]], value)

    if ctype == "daily_forecast_threshold":
        field = condition["field"]
        window = condition.get("window_days", len(weather["daily_forecast"]))
        min_days = condition.get("min_days_triggered", 1)
        days = weather["daily_forecast"][:window]
        return sum(1 for d in days if op(d[field], value)) >= min_days

    return False


# Thứ tự ưu tiên risk_level khi nhiều scenario cùng khớp
RISK_LEVEL_PRIORITY = {
    "high": 0,
    "increased": 1,
    "caution_water_stress": 2,
    "monitoring": 3,
    "normal": 4,
}


def determine_weather_scenario(weather_scenarios: list[dict], weather: dict) -> str:
    """
    Trả về scenario_id phù hợp nhất với dữ liệu thời tiết thật.
    Ưu tiên risk_level cao nhất nếu nhiều scenario cùng khớp.
    Nếu không scenario nào (ngoài WS_NORMAL) khớp -> trả WS_NORMAL.
    """
    if weather is None:
        return "WS_NORMAL"

    matched = []
    for scenario in weather_scenarios:
        if scenario["scenario_id"] == "WS_NORMAL":
            continue
        if _check_condition(scenario["condition"], weather):
            matched.append(scenario)

    if not matched:
        return "WS_NORMAL"

    matched.sort(key=lambda s: RISK_LEVEL_PRIORITY.get(s["risk_level"], 99))
    return matched[0]["scenario_id"]
# ============================================================================
# 3. PROMPT BUILDER
# ============================================================================

def build_prompt(case, source_titles, weather_data=None, location=None):
    ctx = case["llm_context_vi"]["context"]
    instruction = case["llm_context_vi"]["instruction"]
    sources_text = "\n".join(f"• {s}" for s in source_titles)

    weather_summary_section = ""
    weather_guidance_section = ""

    if weather_data is not None:
        summary = format_weather_summary_vi(weather_data, location)
        weather_summary_section = f"""
=========================
THỜI TIẾT KHU VỰC VƯỜN
=========================

{summary}
"""

    tt = ctx.get("thoi_tiet")
    if tt:
        weather_guidance_section = f"""
=========================
KỊCH BẢN THỜI TIẾT ÁP DỤNG: {tt['kich_ban']} (mức rủi ro: {tt['muc_do_rui_ro']})
=========================

{tt['huong_dan']}
"""

    prompt = f"""
Bạn là chuyên gia tư vấn bệnh lá sầu riêng.

{instruction}
{weather_summary_section}{weather_guidance_section}
=========================
THÔNG TIN CHẨN ĐOÁN
=========================

Bệnh:
{ctx['benh']}

Nhãn model:
{ctx['nhan_model']}

Mức độ:
{ctx['muc_do']}

Định nghĩa mức độ:
{ctx['dinh_nghia_muc_do']}

=========================
MÔ TẢ BỆNH
=========================

{ctx['mo_ta_benh']}

=========================
TRIỆU CHỨNG
=========================

{chr(10).join(f"- {x}" for x in ctx['trieu_chung'])}

=========================
ĐIỀU KIỆN NGUY CƠ
=========================

{chr(10).join(f"- {x}" for x in ctx['dieu_kien_nguy_co'])}

=========================
HÀNH ĐỘNG NGAY
=========================

{ctx['hanh_dong_ngay']}

=========================
KHUYẾN NGHỊ QUẢN LÝ
=========================

{chr(10).join(f"- {x}" for x in ctx['khuyen_nghi_quan_ly'])}

=========================
PHÒNG NGỪA
=========================

{chr(10).join(f"- {x}" for x in ctx['phong_ngua'])}

=========================
THEO DÕI
=========================

{chr(10).join(f"- {x}" for x in ctx['theo_doi'])}

=========================
CẢNH BÁO
=========================

{ctx['canh_bao']}

=========================
NGUỒN THAM KHẢO
=========================

{sources_text}

Yêu cầu:

1. Viết bằng tiếng Việt, dễ hiểu cho nông dân.
2. Không bịa thuốc, liều lượng, hoặc số liệu thời tiết ngoài phần "THỜI TIẾT KHU VỰC VƯỜN" đã cho.
3. Trình bày theo cấu trúc:
   - Tóm tắt tình trạng bệnh
   - Tóm tắt thời tiết sắp tới (chỉ dựa trên số liệu đã cho, nếu có)
   - Việc cần làm ngay — NÊU RÕ vì sao (liên hệ trực tiếp giữa thời tiết cụ thể và hành động, ví dụ: "do dự báo mưa X mm vào ngày Y nên cần...")
   - Quản lý
   - Phòng ngừa
   - Theo dõi
   - Cảnh báo
4. Nếu không có dữ liệu thời tiết, bỏ qua phần thời tiết, không tự suy đoán.
5. Cuối cùng liệt kê lại nguồn tham khảo.
"""

    return prompt


# ============================================================================
# 4. GROQ LLM
# ============================================================================

def call_groq(
    prompt: str,
    model: str = "qwen/qwen3.6-27b",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
) -> str:

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please set it in your environment or .env file.")
    client = Groq(api_key=key)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia tư vấn bệnh cây trồng "
                    "và bệnh lá sầu riêng."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return completion.choices[0].message.content


# ============================================================================
# 5. PUBLIC API
# ============================================================================

def get_recommendation(
    disease_idx, severity_idx, kb,
    api_key=None, model="qwen/qwen3.6-27b", max_tokens=1024,
    disease_confidence=None, severity_confidence=None,
    weather_scenario="WS_NORMAL",
    weather_data=None, location=None,
):
    disease_label = DISEASE_IDX_TO_LABEL.get(disease_idx)
    severity_level = SEVERITY_IDX_TO_LEVEL.get(severity_idx)

    if disease_label is None or severity_level is None:
        return {
            "error": f"Index không hợp lệ: disease_idx={disease_idx}, severity_idx={severity_idx}",
            "recommendation_text": None,
        }

    case = kb.lookup(disease_label, severity_level, weather_scenario)

    if case is None:
        return {
            "disease_label": disease_label,
            "severity_level": severity_level,
            "weather_scenario": weather_scenario,
            "error": f"Không tìm thấy case {disease_label} severity={severity_level} weather={weather_scenario}",
            "recommendation_text": None,
        }

    context = case["llm_context_vi"]["context"]
    source_ids = list(context.get("nguon_minh_chung", []))
    thoi_tiet = context.get("thoi_tiet", {})
    if isinstance(thoi_tiet, dict):
        weather_source_ids = thoi_tiet.get("nguon_minh_chung", [])
        for wsid in weather_source_ids:
            if wsid not in source_ids:
                source_ids.append(wsid)
    source_titles = kb.get_source_titles(source_ids)

    prompt = build_prompt(case, source_titles, weather_data=weather_data, location=location)

    try:
        t_start = time.time()
        recommendation_text = call_groq(prompt=prompt, model=model, max_tokens=max_tokens, api_key=api_key)
        groq_time = time.time() - t_start
        error = None
    except Exception as e:
        recommendation_text = None
        error = str(e)
        groq_time = time.time() - t_start

    return {
        "disease_label": disease_label,
        "severity_level": severity_level,
        "weather_scenario": weather_scenario,
        "case_id": case["case_id"],
        "disease_confidence": disease_confidence,
        "severity_confidence": severity_confidence,
        "recommendation_text": recommendation_text,
        "source_titles": source_titles,
        "prompt": prompt,
        "error": error,
        "groq_time": groq_time,
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":

    kb = DurianLeafKB(
        "durian_leaf_kb.json"
    )

    result = get_recommendation(
        disease_idx=1,
        severity_idx=2,
        kb=kb,
        model="qwen/qwen3.6-27b",
    )

    print(result["recommendation_text"])