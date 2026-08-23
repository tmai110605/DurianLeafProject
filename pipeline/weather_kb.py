import requests
import json
from pathlib import Path


def get_location_from_ip() -> dict:
    """Lấy vị trí ước lượng từ IP (fallback nếu user không cung cấp lat/lon)."""
    resp = requests.get("http://ip-api.com/json/", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    return {
        "lat": data["lat"],
        "lon": data["lon"],
        "city": data.get("city"),
        "country": data.get("country"),
    }


def get_weather_forecast(lat: float, lon: float) -> dict:
    """Open-Meteo: hiện tại + dự báo 3 ngày tới, không cần API key."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_max",
        "forecast_days": 3,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    daily = []
    for i in range(len(data["daily"]["time"])):
        daily.append({
            "date": data["daily"]["time"][i],
            "rain_mm": data["daily"]["precipitation_sum"][i],
            "temp_max": data["daily"]["temperature_2m_max"][i],
            "temp_min": data["daily"]["temperature_2m_min"][i],
            "humidity_max": data["daily"]["relative_humidity_2m_max"][i],
        })

    return {
        "current_temp": data["current"]["temperature_2m"],
        "current_humidity": data["current"]["relative_humidity_2m"],
        "current_rain": data["current"]["precipitation"],
        "daily_forecast": daily,
        "rain_mm_3day_total": sum(d["rain_mm"] for d in daily),
    }



def format_weather_summary_vi(weather: dict, location: dict = None) -> str:
    """Mô tả thời tiết hiện tại + dự báo bằng tiếng Việt, dùng số liệu thật từ API."""
    lines = []

    if location:
        loc_name = location.get("city") or f"{location['lat']:.3f}, {location['lon']:.3f}"
        lines.append(f"Vị trí: {loc_name}")

    lines.append(
        f"Hiện tại: {weather['current_temp']}°C, "
        f"độ ẩm {weather['current_humidity']}%, "
        f"lượng mưa {weather['current_rain']}mm"
    )

    lines.append("Dự báo 3 ngày tới:")
    for d in weather["daily_forecast"]:
        lines.append(
            f"- {d['date']}: nhiệt độ {d['temp_min']}–{d['temp_max']}°C, "
            f"độ ẩm tối đa {d['humidity_max']}%, "
            f"lượng mưa {d['rain_mm']}mm"
        )

    lines.append(
        f"Tổng lượng mưa dự kiến 3 ngày tới: {weather['rain_mm_3day_total']:.1f}mm"
    )

    return "\n".join(lines)


def get_address_from_coords(lat: float, lon: float) -> dict:
    """Sử dụng OpenStreetMap Nominatim để lấy địa chỉ chi tiết từ tọa độ GPS."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        headers = {
            "User-Agent": "DurianLeafDisease/1.0 (contact: support@durianleaf.com)"
        }
        params = {
            "format": "json",
            "lat": lat,
            "lon": lon,
            "zoom": 18,
            "addressdetails": 1
        }
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        address = data.get("address", {})
        # Lấy các thành phần địa chỉ chi tiết nhất có thể
        road = address.get("road")
        suburb = address.get("suburb") or address.get("quarter") or address.get("neighbourhood")
        district = address.get("county") or address.get("district") or address.get("city_district")
        city = address.get("city") or address.get("town") or address.get("village") or address.get("state")
        country = address.get("country")

        parts = [p for p in [road, suburb, district, city] if p]
        detailed_address = ", ".join(parts) if parts else data.get("display_name")

        return {
            "lat": lat,
            "lon": lon,
            "city": detailed_address or city or f"{lat:.4f}, {lon:.4f}",
            "country": country or "Vietnam"
        }
    except Exception as e:
        print(f"Reverse geocoding failed: {e}")
        return {
            "lat": lat,
            "lon": lon,
            "city": f"{lat:.4f}, {lon:.4f}",
            "country": "Vietnam"
        }