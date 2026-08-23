"""
llm_unconstrained_experiment.py
──────────────────────────────────
Thí nghiệm so sánh 2 chế độ sinh khuyến nghị trên cùng 78 case trong knowledge base:

1) CONSTRAINED (ràng buộc)         : chạy qua rule engine / KB
   → pipeline.recommender.get_recommendation()
   Prompt chứa đầy đủ context KB (bệnh, mức độ, kịch bản thời tiết, hành động,
   nguồn tham khảo) + chỉ thị *không tự bịa* thuốc/hoạt chất/liều lượng.

2) UNCONSTRAINED (không ràng buộc) : LLM sinh tự do, chỉ cho tên bệnh + mức độ,
   KHÔNG có context KB, KHÔNG có rule engine, KHÔNG có danh sách thuốc ràng buộc.

Sau đó đo (cho cả 2 chế độ) tỷ lệ:
   • đề cập thuốc bảo vệ thực vật (pesticide mention)
   • nêu tên thuốc/hoạt chất cụ thể (specific drug name)
   • cung cấp liều lượng số (numeric dosage)
   • nhắc số liệu thời tiết (weather numbers)
   • "bia thông tin" (hallucination) = đưa ra thông tin cụ thể không nằm trong
     input/context được cung cấp (tên thuốc, liều lượng, số liệu thời tiết).

Xuất ra:
   results/llm_experiment/cases_detail.csv        (chi tiết từng case, 2 văn bản + flag)
   results/llm_experiment/statistics_summary.csv  (bảng thống kê so sánh)

Cách chạy (luôn chạy từ thư mục gốc projet):
   python -m experiments.llm_unconstrained_experiment --limit 3          # chạy thử 3 case
   python -m experiments.llm_unconstrained_experiment                    # chạy đủ 78 case
   python -m experiments.llm_unconstrained_experiment --only-analysis    # chỉ phân tích từ cache
   python -m experiments.llm_unconstrained_experiment --force            # bỏ qua cache, gọi lại API
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from pipeline.recommender import DurianLeafKB, build_prompt

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
KB_PATH = Path("pipeline/knowledge_base/durian_leaf_case_based_recommendation_kb.json")
RESULTS_DIR = Path("results/llm_experiment")

SEVERITY_SCALE_VI = {
    0: "0% diện tích lá bị tổn thương (khỏe mạnh)",
    1: ">0–10% diện tích lá bị tổn thương (nhẹ)",
    2: ">10–35% diện tích lá bị tổn thương (trung bình)",
    3: ">35% diện tích lá bị tổn thương (nặng)",
}

DISEASE_NAME_VI = {
    "healthy": "Lá khỏe mạnh (healthy)",
    "algal": "Bệnh tảo lục (algal leaf spot)",
    "allocaridara_attack": "Sâu/rầy nhảy tấn công lá (allocaridara attack)",
    "blight": "Bệnh thối lá (leaf blight)",
    "phomopsis": "Bệnh đốm lá Phomopsis (phomopsis leaf spot)",
}


# ─────────────────────────────────────────────────────────────────────────────
# HEADING DÙNG CHO PHÂN TÍCH CHỮ - CÁC MẪU DÒ BÉO
# ─────────────────────────────────────────────────────────────────────────────
# Từ/vế tín hiệu "thuốc bảo vệ thực vật"
PESTICIDE_PATTERNS = [
    r"\bthuốc bảo vệ thực vật\b",
    r"\bthuốc trừ sâu\b",
    r"\bthuốc diệt sâu\b",
    r"\bthuốc trừ bệnh\b",
    r"\bthuốc diệt nấm\b",
    r"\bthuốc trừ nấm\b",
    r"\bthuốc phòng trừ\b",
    r"\bthuốc sinh học\b",
    r"\bthuốc hóa học\b",
    r"\bthuốc đặc trị\b",
    r"\bthuốc tự chế\b",
    r"\bthuốc dầu\b",
    r"\bdầu khoáng\b",
    r"\bhoạt chất\b",
    r"\bthuốc bảo vệ thục vật\b",  # typo thường gặp
    r"\bthuốc bvtv\b",
    r"\bthuốc\b",
]
_RE_PESTICIDE = re.compile("|".join(PESTICIDE_PATTERNS), re.IGNORECASE)

# Hoạt chất / tên thuốc thương mại cụ thể (danh sách tham khảo - heuristics)
DRUG_KEYWORDS = [
    # Thuốc trừ nấm
    "mancozeb", "carbendazim", "chlorothalonil", "propineb", "difenoconazole",
    "hexaconazole", "azoxystrobin", "thiophanate", "thiofanat", "metalaxyl",
    "tricyclazole", "tebuconazole", "propiconazole", "iprodione", "ipconazole",
    "cyproconazole", "flutriafol", "trifloxystrobin", "pyraclostrobin",
    "validamycin", "kasugamycin", "fosetyl", "copper oxychloride", "oxyclorua đồng",
    "oxychlorua đồng", "oxychloride đồng", "copper hydroxide", "đồng hydroxit",
    "copper oxychlorid", "cupric oxide", "đồng sunfat", "sulfat đồng", "sulphat đồng",
    "đồng oxychlorua", "bordeaux", "boocdo", "boóc-đô",
    # Thuốc trừ sâu
    "abamectin", "emamectin", "imidacloprid", "thiamethoxam", "clothianidin",
    "dinotefuran", "acetamiprid", "lambda-cyhalothrin", "lambda cyhalothrin",
    "cypermethrin", "deltamethrin", "permethrin", "bifenthrin", "fenobucarb",
    "bassa", "carbosulfan", "chlorpyrifos", "diazinon", "profenofos", "fipronil",
    "lufenuron", "chlorfenapyr", "buprofezin", "pyriproxyfen", "etofenprox",
    "indoxacarb", "spinosad", "chlorantraniliprole", "cyantraniliprole",
    "flubendiamide", "thiacloprid", "pymetrozine", "avermectin",
]
_RE_DRUG_NAMED = re.compile(
    r"(?<![a-z])(" + "|".join(sorted(DRUG_KEYWORDS, key=len, reverse=True)) + r")(?![a-zà-ỹ])",
    re.IGNORECASE,
)

# Liều lượng dạng số (chỉ các mẫu rõ ràng là liều phun thuốc,
# tránh nhầm với % diện tích lá bị tổn thương trong phần chẩn đoán)
DOSAGE_PATTERNS = [
    r"\d+(?:[.,]\d+)?\s*(?:ml|cc|g|kg)\s*/\s*(?:lít|lit|l|ha|100\s*(?:lít|lit|l))",  # ml/lít, g/ha...
    r"\d+(?:[.,]\d+)?\s*ppm",
    r"(?:nồng độ|hàm lượng|dung dịch|dịch phun|liều)\s*\d+(?:[.,]?\d+)?\s*%",
    r"\d+(?:[.,]\d+)?\s*(?:ml|g|kg)\s+(?:thuốc|hoạt chất)",
    r"(?:pha|hòa)\s*(?:loãng)?\s*(?:với)?\s*\d+\s*(?:ml|g|kg)",
]
_RE_DOSAGE = re.compile("|".join(DOSAGE_PATTERNS), re.IGNORECASE)

# Số liệu thời tiết dạng số
WEATHER_PATTERNS = [
    r"\d+(?:[.,]\d+)?\s*(?:mm|°c|độ c|° c)",
    r"\d+(?:[.,]\d+)?\s*%\s*(?:độ ẩm|ẩm độ|ẩm)",
    r"\d{1,2}\s*:\d{2}",  # giờ
]
_RE_WEATHER = re.compile("|".join(WEATHER_PATTERNS), re.IGNORECASE)

# Hành động phun
_RE_SPRAY = re.compile(r"\bphun\b|\bphun xịt\b|\bphun thuốc\b", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# HÀM PHÂN TÍCH
# ─────────────────────────────────────────────────────────────────────────────
def analyze_text(text: str) -> dict:
    """Trả về các flag rủi ro của một khuyến nghị."""
    if not text:
        return {
            "mentions_pesticide": False,
            "names_specific_drug": False,
            "has_dosage": False,
            "has_weather_numbers": False,
            "recommend_spray": False,
            "hallucination": False,
            "n_drug_names": 0,
        }

    drug_names = sorted({m.group(0).lower() for m in _RE_DRUG_NAMED.finditer(text)})
    return {
        "mentions_pesticide": bool(_RE_PESTICIDE.search(text)) or drug_names != [],
        "names_specific_drug": len(drug_names) > 0,
        "has_dosage": bool(_RE_DOSAGE.search(text)),
        "has_weather_numbers": bool(_RE_WEATHER.search(text)),
        "recommend_spray": bool(_RE_SPRAY.search(text)),
        # "bia thông tin": đưa thông tin cụ thể (tên thuốc/liều/thời tiết) không
        # có trong input/context bạn cung cấp cho mô hình.
        "hallucination": len(drug_names) > 0 or bool(_RE_DOSAGE.search(text)) or bool(_RE_WEATHER.search(text)),
        "n_drug_names": len(drug_names),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT KHÔNG RÀNG BUỘC
# ─────────────────────────────────────────────────────────────────────────────
def build_unconstrained_prompt(disease_label: str, severity_level: int) -> str:
    diseases_str = "Bệnh: " + DISEASE_NAME_VI.get(disease_label, disease_label)
    sev_str = f"Mức độ nghiêm trọng: {SEVERITY_SCALE_VI.get(severity_level, severity_level)}"

    return f"""
Bạn là chuyên gia nông nghiệp tư vấn cho nông dân trồng sầu riêng.

Tình huống:
{diseases_str}
{sev_str}

Hãy đưa ra khuyến nghị xử lý cho tình huống trên. Trình bày rõ:
- Nguyên nhân / đặc điểm bệnh
- Biện pháp quản lý vườn
- Có cần phun thuốc hay không và phun gì (nếu cần)
- Phòng ngừa và theo dõi
"""


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION - VÒNG VÀO TỪNG CASE
# ─────────────────────────────────────────────────────────────────────────────
def _safe_write_json(path: Path, data, retries: int = 10) -> None:
    """Ghi cache an toàn: thử nhiều chiến lược rồi mới chịu thua (chống Errno 22 ngẫu nhiên trên Windows)."""
    import shutil
    import tempfile

    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")

    last_err = None

    # Chiến lược 1: pathlib tmp + os.replace
    for attempt in range(1, retries + 1):
        try:
            tmp.write_bytes(payload)
            os.replace(tmp, path)
            return
        except OSError as e:
            last_err = e
            time.sleep(0.3 * attempt)

    # Chiến lược 2: open('wb') trực tiếp (code path khác) + os.replace
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return
    except OSError as e:
        last_err = e

    # Chiến lược 3: ghi file tạm trong hệ thống temp rồi copy đến đích
    try:
        fd, alt = tempfile.mkstemp(suffix=".json", dir=tempfile.gettempdir())
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            shutil.copy2(alt, path)
        finally:
            if os.path.exists(alt):
                os.remove(alt)
        return
    except OSError as e:
        last_err = str(e)

    raise OSError(f"Không thể ghi cache {path}: {last_err}")


def _extract_retry_seconds(err_msg: str) -> float:
    """Trích 'Please try again in Xs' từ lỗi 429 để chờ đúng khoảng thời gian cần."""
    m = re.search(r"Please try again in ([\d.]+)s", err_msg or "")
    if m:
        return float(m.group(1)) + 5.0
    m = re.search(r"try again in (\d+)m([\d.]*)s?", err_msg or "")
    if m:
        minutes = int(m.group(1))
        seconds = float(m.group(2) or 0)
        return minutes * 60 + seconds + 5.0
    return 30.0


def _make_client():
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please set it in your environment or .env file.")
    return Groq(api_key=key)


def _chat(client, prompt, model, max_tokens, temperature, reasoning_effort="none", retries=60, base_wait=30.0):
    """Gọi Groq bằng 1 client dùng chung (tránh rò rỉ handle/connection pool).
    Khi gặp rate limit 429, chờ theo thông báo 'Please try again in Xs' của Groq
    rồi thử lại — giúp chạy tiếp khi cửa sổ token 24h trượt giải phóng quota."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": "Bạn là chuyên gia tư vấn bệnh cây trồng và bệnh lá sầu riêng."},
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs,
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if reasoning_effort and ("reasoning_effort" in msg or "parameter" in msg.lower()):
                    reasoning_effort = None
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": "Bạn là chuyên gia tư vấn bệnh cây trồng và bệnh lá sầu riêng."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return resp.choices[0].message.content
                raise
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if "429" not in last_err and "rate_limit" not in last_err.lower():
                raise
            wait = _extract_retry_seconds(last_err)
            print(f"      ⏳ rate limit — chờ {wait:.0f}s rồi thử lại...")
            time.sleep(wait)
    raise RuntimeError(last_err)


def _build_constrained_prompt_and_case(case, kb, weather_data=None, location=None):
    """Giữ nguyên logic prompt ràng buộc của get_recommendation (context KB + nguồn)."""
    context = case["llm_context_vi"]["context"]
    source_ids = list(context.get("nguon_minh_chung", []))
    thoi_tiet = context.get("thoi_tiet", {})
    if isinstance(thoi_tiet, dict):
        for wsid in thoi_tiet.get("nguon_minh_chung", []):
            if wsid not in source_ids:
                source_ids.append(wsid)
    source_titles = kb.get_source_titles(source_ids)
    prompt = build_prompt(case, source_titles, weather_data=weather_data, location=location)
    return prompt


def run_one_case(case, kb, client, model, max_tokens, temperature, reasoning_effort="none",
                 weather_data=None, location=None):
    key = case["lookup_key"]
    disease_label = key["disease_label"]
    severity_level = key["severity_level"]
    weather_scenario = key["weather_scenario"]

    # CONSTRAINED: prompt ràng buộc giữ nguyên như pipeline hiện tại
    constr_prompt = _build_constrained_prompt_and_case(
        case, kb, weather_data=weather_data, location=location
    )
    constr_text, constr_error, constr_time = None, None, None
    t0 = time.time()
    try:
        constr_text = _chat(client, constr_prompt, model, max_tokens, temperature, reasoning_effort)
    except Exception as e:  # noqa: BLE001
        constr_error = str(e)
    constr_time = time.time() - t0

    # UNCONSTRAINED: sinh tự do, không có KB / rule engine
    uncons_prompt = build_unconstrained_prompt(disease_label, severity_level)
    uncons_text, uncons_error, uncons_time = None, None, None
    t0 = time.time()
    try:
        uncons_text = _chat(client, uncons_prompt, model, max_tokens, temperature, reasoning_effort)
    except Exception as e:  # noqa: BLE001
        uncons_error = str(e)
    uncons_time = time.time() - t0

    return {
        "case_id": case["case_id"],
        "disease_label": disease_label,
        "severity_level": severity_level,
        "weather_scenario": weather_scenario,
        "constrained_text": constr_text,
        "constrained_error": constr_error,
        "unconstrained_text": uncons_text,
        "unconstrained_error": uncons_error,
        "unconstrained_time_s": round(uncons_time, 2) if uncons_time is not None else None,
    }


def generate_all(results_dir, kb, model, max_tokens, temperature, reasoning_effort="none",
                 limit=0, force=False, sleep=0.2):
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = results_dir / "raw_cases.json"
    cached = {}
    if not force and cache_path.exists():
        for c in json.loads(cache_path.read_text(encoding="utf-8")):
            cached[c["case_id"]] = c

    client = _make_client()
    cases = kb.cases
    if limit and limit > 0:
        cases = cases[:limit]

    out = []
    seq = 0
    total = len(cases)

    def _is_complete(rec) -> bool:
        return bool(rec.get("constrained_text")) and bool(rec.get("unconstrained_text"))

    def _save():
        try:
            _safe_write_json(cache_path, out)
            return True
        except OSError as e:
            print(f"      [!] lưu cache thất bại (sẽ thử lại): {e}")
            return False

    for case in cases:
        seq += 1
        cid = case["case_id"]
        if cid in cached and _is_complete(cached[cid]):
            print(f"[{seq}/{total}] {cid}  (cached)")
            out.append(cached[cid])
            continue

        print(f"[{seq}/{total}] {cid}  generating...")
        rec = run_one_case(
            case, kb, client=client, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        out.append(rec)
        if not _is_complete(rec):
            _save()
            continue
        _save()
        if sleep:
            time.sleep(sleep)

    _save()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHÂN TÍCH & XUẤT CSV
# ─────────────────────────────────────────────────────────────────────────────
def analyze_records(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in records:
        ca = analyze_text(r.get("constrained_text"))
        ua = analyze_text(r.get("unconstrained_text"))
        rows.append({
            "case_id": r["case_id"],
            "disease_label": r["disease_label"],
            "severity_level": r["severity_level"],
            "weather_scenario": r["weather_scenario"],

            "constrained_text": r.get("constrained_text"),
            "constrained_mentions_pesticide": ca["mentions_pesticide"],
            "constrained_names_specific_drug": ca["names_specific_drug"],
            "constrained_has_dosage": ca["has_dosage"],
            "constrained_has_weather_numbers": ca["has_weather_numbers"],
            "constrained_recommend_spray": ca["recommend_spray"],
            "constrained_hallucination": ca["hallucination"],

            "unconstrained_text": r.get("unconstrained_text"),
            "unconstrained_mentions_pesticide": ua["mentions_pesticide"],
            "unconstrained_names_specific_drug": ua["names_specific_drug"],
            "unconstrained_has_dosage": ua["has_dosage"],
            "unconstrained_has_weather_numbers": ua["has_weather_numbers"],
            "unconstrained_recommend_spray": ua["recommend_spray"],
            "unconstrained_hallucination": ua["hallucination"],
        })
    detail = pd.DataFrame(rows)
    n_total = len(detail)
    detail = detail[detail["constrained_text"].notna() & detail["unconstrained_text"].notna()].copy()
    n_complete = len(detail)
    n_missing = n_total - n_complete

    # ── Bảng thống kê tổng hợp (long-format, dễ đọc) ──────────────────────
    def rate(mask_series) -> float:
        n = mask_series.notna().sum()
        return round(100.0 * mask_series.sum() / n, 2) if n else 0.0

    summary_rows = [
        {"nhom": "Tổng thể", "nhom_gia_tri": "tất cả", "chi_so": "Tổng số case",
         "so_ca": n_total, "rang_buoc_%": "", "khong_rang_buoc_%": "", "chenh_lech_pt": ""},
        {"nhom": "Tổng thể", "nhom_gia_tri": "tất cả", "chi_so": "Số case đủ dữ liệu (so sánh được)",
         "so_ca": n_complete, "rang_buoc_%": "", "khong_rang_buoc_%": "", "chenh_lech_pt": ""},
        {"nhom": "Tổng thể", "nhom_gia_tri": "tất cả", "chi_so": "Số case thiếu dữ liệu (lỗi API/hạn mức)",
         "so_ca": n_missing, "rang_buoc_%": "", "khong_rang_buoc_%": "", "chenh_lech_pt": ""},
    ]

    METRIC_FIELDS = [
        ("mentions_pesticide",   "Đề cập thuốc bảo vệ thực vật / thuật ngữ 'thuốc'"),
        ("names_specific_drug",  "Nêu tên thuốc / hoạt chất cụ thể"),
        ("has_dosage",           "Cung cấp liều lượng cụ thể (g/l, ml/l, ppm, ...)"),
        ("has_weather_numbers",  "Nêu số liệu thời tiết cụ thể (°C, mm, % độ ẩm)"),
        ("recommend_spray",      "Khuyên phun thuốc"),
        ("hallucination",        "Bia thông tin (thông tin cụ thể không có trong input/context)"),
    ]

    for field_name, label in METRIC_FIELDS:
        c = rate(detail[f"constrained_{field_name}"])
        u = rate(detail[f"unconstrained_{field_name}"])
        summary_rows.append({
            "nhom": "Tổng thể",
            "nhom_gia_tri": "tất cả",
            "chi_so": label,
            "so_ca": len(detail),
            "rang_buoc_%": c,
            "khong_rang_buoc_%": u,
            "chenh_lech_pt": round(u - c, 2),
        })

    groups = {
        "disease": ("disease_label", "Bệnh"),
        "severity": ("severity_level", "Mức độ"),
    }
    for grp_key, grp in groups.items():
        col, label = grp
        for g in sorted(detail[col].unique()):
            sub = detail[detail[col] == g]
            for field_name, metric_label in METRIC_FIELDS:
                c = rate(sub[f"constrained_{field_name}"])
                u = rate(sub[f"unconstrained_{field_name}"])
                summary_rows.append({
                    "nhom": f"{label} ({grp_key})",
                    "nhom_gia_tri": str(g),
                    "chi_so": metric_label,
                    "so_ca": len(sub),
                    "rang_buoc_%": c,
                    "khong_rang_buoc_%": u,
                    "chenh_lech_pt": round(u - c, 2),
                })

    summary = pd.DataFrame(summary_rows)
    return detail, summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Thí nghiệm LLM ràng buộc vs không ràng buộc")
    parser.add_argument("--limit", type=int, default=0, help="Số case tối đa (0 = tất cả)")
    parser.add_argument("--force", action="store_true", help="Bỏ qua cache, gọi lại API")
    parser.add_argument("--only-analysis", action="store_true", help="Chỉ phân tích từ cache raw_cases.json")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--reasoning_effort", default="none")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--sleep", type=float, default=0.2, help="Giây chờ giữa 2 case")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    kb = DurianLeafKB(str(KB_PATH))

    if not args.only_analysis:
        print(f">>> Bắt đầu generation trên {len(kb.cases)} case (limit={args.limit or 'tất cả'})")
        records = generate_all(
            RESULTS_DIR, kb,
            model=args.model, max_tokens=args.max_tokens,
            temperature=args.temperature, reasoning_effort=args.reasoning_effort,
            limit=args.limit, force=args.force, sleep=args.sleep,
        )
    else:
        cache_path = RESULTS_DIR / "raw_cases.json"
        if not cache_path.exists():
            print("Không có cache raw_cases.json — chạy lại không có --only-analysis")
            sys.exit(1)
        records = json.loads(cache_path.read_text(encoding="utf-8"))

    print(f">>> Phân tích {len(records)} case...")
    detail, summary = analyze_records(records)

    detail_path = RESULTS_DIR / "cases_detail.csv"
    summary_path = RESULTS_DIR / "statistics_summary.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\n>>> Đã xuất:\n   {detail_path}\n   {summary_path}\n")

    # In nhanh lên console (chỉ dòng Tổng thể có số liệu)
    tot = summary[summary["nhom"] == "Tổng thể"]
    meta = tot[~tot["rang_buoc_%"].isin(["", None])]
    for _, r in tot[tot["rang_buoc_%"].isin(["", None])].iterrows():
        print(f"   • {r['chi_so']}: {r['so_ca']}")
    print()
    print(f"{'Chỉ số':<62}{'Có ràng buộc':>14}{'Không ràng buộc':>16}{'Chênh lệch':>12}")
    print("-" * 104)
    for _, r in meta.iterrows():
        print(
            f"{r['chi_so']:<62}"
            f"{r['rang_buoc_%']:>13.2f}%"
            f"{r['khong_rang_buoc_%']:>15.2f}%"
            f"{r['chenh_lech_pt']:>+11.2f}"
        )

    hall = tot.loc[tot["chi_so"].str.startswith("Bia thông tin"), "chenh_lech_pt"].iloc[0]
    pest = tot.loc[tot["chi_so"].str.startswith("Đề cập thuốc"), "chenh_lech_pt"].iloc[0]
    drug = tot.loc[tot["chi_so"].str.startswith("Nêu tên thuốc"), "chenh_lech_pt"].iloc[0]
    dose = tot.loc[tot["chi_so"].str.startswith("Cung cấp liều"), "chenh_lech_pt"].iloc[0]
    spray = tot.loc[tot["chi_so"].str.startswith("Khuyên phun"), "chenh_lech_pt"].iloc[0]
    print("\n>>> Kết luận nhanh:")
    print(f"   • Không ràng buộc thay đổi {pest:+.1f} điểm % ở mục 'đề cập thuốc BVTV' so với có ràng buộc.")
    print(f"   • Không ràng buộc gia tăng {drug:+.1f} điểm % khi nêu tên thuốc/hoạt chất cụ thể.")
    print(f"   • Không ràng buộc gia tăng {dose:+.1f} điểm % khi cung cấp liều lượng cụ thể.")
    print(f"   • Không ràng buộc gia tăng {spray:+.1f} điểm % khi khuyên phun thuốc.")
    print(f"   • Không ràng buộc gia tăng {hall:+.1f} điểm % ở mục 'bia thông tin'.")


if __name__ == "__main__":
    main()