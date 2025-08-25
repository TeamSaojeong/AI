import os
import json
import glob
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# --- Optional GPU (XGBoost) ---
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# --- Optional API calls ---
try:
    import requests
    REQ_AVAILABLE = True
except Exception:
    REQ_AVAILABLE = False


# =========================
# 설정값 / 규칙 가중치
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.2
RF_ESTIMATORS = 300

XGB_PARAMS = {
    "max_depth": 7,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "eval_metric": "mlogloss",
    "n_estimators": 400,
    "verbosity": 0,
    "random_state": RANDOM_STATE,
}

# 구역 유형별 시간 가중
RULE_WEIGHTS = {
    "OFFICE": {
        "boost": [("06:00", "09:00")],      # 출근 ↑
        "suppress": [("16:30", "21:00")],   # 퇴근 ↓
        "factor_up": 1.20,
        "factor_down": 0.85,
    },
    "ENTERTAINMENT": {
        "boost": [("10:00", "11:00"), ("18:00", "21:00")],
        "suppress": [],
        "factor_up": 1.30,
        "factor_down": 1.00,
    },
    "MIXED": {
        "boost": [],
        "suppress": [],
        "factor_up": 1.05,
        "factor_down": 1.00,
    },
}

# 주말(토/일) 10~18 혼잡 ↑
WEEKEND_RULE = {
    "window": ("10:00", "18:00"),
    "factor_up": 1.20,
}

# 등급 표준화 (원본 → 3단계)
LEVEL_NORMALIZE = {
    "여유": "여유",
    "보통": "보통",
    "약간 혼잡": "혼잡",
    "약간 붐빔": "혼잡",
    "붐빔": "혼잡",
    "혼잡": "혼잡",
    "매우 혼잡": "혼잡",
    "매우 붐빔": "혼잡",
}
LEVEL_ORDER = ["여유", "보통", "혼잡"]

# ===== 장소 타입 매핑 (Google Places types → 내부 facility_tag) =====
ENTERTAINMENT_TYPES = {
    "shopping_mall", "department_store", "movie_theater", "amusement_park",
    "aquarium", "art_gallery", "museum", "zoo", "stadium", "tourist_attraction",
    "casino", "bowling_alley", "spa", "night_club", "book_store", "convention_center"
}
OFFICE_TYPES = {
    "university", "school", "bank", "embassy", "courthouse", "city_hall",
    "local_government_office", "post_office", "train_station",
    "subway_station", "bus_station"
}
PARK_TYPES = {"park", "campground"}

# 문화/쇼핑 주말 강화 옵션
CULTURE_WEEKEND_FACTOR_UP = 1.6
CULTURE_WEEKEND_MIN_PROB = 0.55
CULTURE_WEEKEND_WINDOW = ("10:00", "18:00")


# =========================
# 유틸
# =========================
def _hhmm_to_float(s: str) -> float:
    h, m = s.split(":")
    return int(h) + int(m) / 60.0


def in_any_window(hour_float: float, windows: List[Tuple[str, str]]) -> bool:
    if not windows:
        return False
    for s, e in windows:
        if _hhmm_to_float(s) <= hour_float < _hhmm_to_float(e):
            return True
    return False


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def normalize_level(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t in LEVEL_NORMALIZE:
        return LEVEL_NORMALIZE[t]
    if "여유" in t or "붐빔은 거의" in t:
        return "여유"
    if "보통" in t or "무난" in t:
        return "보통"
    if "혼잡" in t or "붐빔" in t or "몰려" in t:
        return "혼잡"
    return None


def get_class_index(classes: np.ndarray, name: str) -> Optional[int]:
    for i, c in enumerate(classes):
        if c == name:
            return i
    return None


# =========================
# 데이터 로드 / 전처리
# =========================
def load_and_concat(files_glob: str) -> pd.DataFrame:
    files = glob.glob(files_glob)
    if not files:
        raise FileNotFoundError(f"No CSV files matched pattern: {files_glob}")
    dfs = []
    for p in sorted(files):
        try:
            df = pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="cp949")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    required = ["areaCd", "collectedAt", "areaCongestLvl", "prkCd", "prkNm", "lng", "lat"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    if "prkType" not in df.columns:
        df["prkType"] = "UNK"

    df["collectedAt"] = pd.to_datetime(df["collectedAt"], errors="coerce")
    df = df.dropna(subset=["collectedAt", "areaCd", "prkCd", "lng", "lat"])
    df["hour"] = df["collectedAt"].dt.hour
    df["minute"] = df["collectedAt"].dt.minute
    df["dow"] = df["collectedAt"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    df["level_std"] = df["areaCongestLvl"].astype(str).map(normalize_level)
    df = df.dropna(subset=["level_std"])

    for c in ["lng", "lat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lng", "lat"])

    for c in ["temp", "humidity", "pm10"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    return df


def build_lots_meta(df: pd.DataFrame) -> pd.DataFrame:
    """주차장별(lat/lng/areaCd/prkType/prkNm) 최신 메타(가장 최근 레코드)."""
    df = df.sort_values("collectedAt")
    meta = df.groupby("prkCd").tail(1)[["prkCd", "prkNm", "areaCd", "prkType", "lat", "lng"]].copy()
    meta = meta.dropna(subset=["lat", "lng"])
    meta["prkType"] = meta["prkType"].fillna("UNK").astype(str)
    return meta.reset_index(drop=True)


def infer_area_type(df: pd.DataFrame) -> Dict[str, str]:
    """구역별 시간대 분포로 유형 추정."""
    prof = df.groupby(["areaCd", "hour"])["level_std"].count().unstack(fill_value=0)
    prof = prof.div(prof.max(axis=1).replace(0, np.nan), axis=0).fillna(0)
    types = {}
    for area, row in prof.iterrows():
        h_peak = int(np.argmax(row.values))
        if 6 <= h_peak <= 9:
            types[area] = "OFFICE"
        elif h_peak in [10, 11] or 18 <= h_peak <= 21:
            types[area] = "ENTERTAINMENT"
        else:
            types[area] = "MIXED"
    return types


def prepare_features(df: pd.DataFrame):
    """라벨 인코딩 + one-hot 스키마 생성 (DataFrame 유지 → feature name 경고 방지)."""
    le = LabelEncoder()
    df["y"] = le.fit_transform(df["level_std"])
    df_oh = pd.get_dummies(df, columns=["areaCd", "prkType"], drop_first=False)
    base_feats = ["hour", "minute", "dow", "is_weekend", "temp", "humidity", "pm10"]
    oh_feats = [c for c in df_oh.columns if c.startswith("areaCd_") or c.startswith("prkType_")]
    Xcols = base_feats + oh_feats
    X = df_oh[Xcols].fillna(0.0).astype(np.float32)
    y = df_oh["y"].values
    return X, y, Xcols, le


# =========================
# API 유틸 (캐시 포함)
# =========================
def _read_key(env_name: str, cli_value: Optional[str]) -> Optional[str]:
    return cli_value or os.environ.get(env_name)


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(path: str, obj: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def fetch_place_types_google(lat: float, lon: float, radius_m: int, api_key: str, model_dir: str) -> List[str]:
    """좌표 주변의 장소 types(상위 몇 개)만 가벼운 컨텍스트로 사용."""
    if not REQ_AVAILABLE:
        return []
    cache_path = os.path.join(model_dir, "places_ctx_cache.json")
    cache = _load_cache(cache_path)
    key = f"{round(lat,5)}:{round(lon,5)}:{radius_m}"
    if key in cache:
        return cache[key]

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"location": f"{lat},{lon}", "radius": radius_m, "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        types_all = []
        for res in (data.get("results") or [])[:8]:
            types_all.extend(res.get("types") or [])
        types_all = list(dict.fromkeys(types_all))
        cache[key] = types_all
        _save_cache(cache_path, cache)
        return types_all
    except Exception:
        return []


def map_types_to_facility_tag(types_list: List[str]) -> Optional[str]:
    if not types_list:
        return None
    s = set(types_list)
    if s & ENTERTAINMENT_TYPES:
        return "CULTURE"
    if s & PARK_TYPES:
        return "PARK"
    if s & OFFICE_TYPES:
        return "OFFICE"
    return None


def fetch_weather_owm(lat: float, lon: float, dt: pd.Timestamp, api_key: str, model_dir: str) -> dict:
    """OpenWeatherMap One Call 3.0 시간별에서 도착시각과 가장 가까운 값."""
    if not REQ_AVAILABLE:
        return {"temp": None, "main": "", "is_precip": False, "precip_mm": 0.0}
    cache_path = os.path.join(model_dir, "weather_cache.json")
    cache = _load_cache(cache_path)
    key = f"{round(lat,3)}:{round(lon,3)}:{dt.strftime('%Y%m%d%H')}"
    if key in cache:
        return cache[key]

    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {"lat": lat, "lon": lon, "units": "metric", "exclude": "minutely,daily,alerts", "appid": api_key}
    try:
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly") or []
        if hourly:
            target = int(dt.timestamp())
            cand = min(hourly, key=lambda h: abs((h.get("dt", 0) - target)))
            temp = float(cand.get("temp", 0.0))
            weather = (cand.get("weather") or [{}])[0]
            main = str(weather.get("main", ""))
            rain = cand.get("rain", {}).get("1h", 0.0)
            snow = cand.get("snow", {}).get("1h", 0.0)
            precip = float(rain or 0.0) + float(snow or 0.0)
            out = {"temp": temp, "main": main, "is_precip": precip > 0.05, "precip_mm": precip}
        else:
            out = {"temp": None, "main": "", "is_precip": False, "precip_mm": 0.0}
        cache[key] = out
        _save_cache(cache_path, cache)
        return out
    except Exception:
        return {"temp": None, "main": "", "is_precip": False, "precip_mm": 0.0}


def discover_parking_via_places(lat: float, lon: float, radius_m: int, api_key: str, want_k: int) -> pd.DataFrame:
    """Google Places Nearby(type=parking)로 반경 내 주차장 탐색 → DataFrame(prkCd/Name/lat/lng/types)."""
    if not REQ_AVAILABLE:
        return pd.DataFrame()
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"location": f"{lat},{lon}", "radius": radius_m, "type": "parking", "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        rows = []
        for res in (data.get("results") or [])[: max(20, want_k)]:
            place_id = res.get("place_id") or res.get("reference")
            name = res.get("name", "주차장")
            geo = res.get("geometry", {}).get("location", {})
            plat, plng = float(geo.get("lat", 0)), float(geo.get("lng", 0))
            types = res.get("types") or []
            rows.append({
                "prkCd": f"EXT_{place_id}",
                "prkNm": name,
                "areaCd": "EXT",
                "prkType": "EXT",
                "lat": plat,
                "lng": plng,
                "types": types
            })
        df = pd.DataFrame(rows)
        if len(df):
            df["dist_km"] = df.apply(lambda row: haversine_km(lat, lon, float(row["lat"]), float(row["lng"])), axis=1)
            df = df.sort_values("dist_km").head(want_k).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# =========================
# 규칙 가중치 / 날씨/시설 반영
# =========================
def _enforce_min_prob_for_class(p: np.ndarray, idx: int, min_prob: float) -> np.ndarray:
    """특정 클래스 확률을 최소 보장하면서 나머지를 비례 축소."""
    p = p.copy()
    total = p.sum()
    if total <= 0:
        p[idx] = 1.0
        return p
    if p[idx] >= min_prob:
        return p / total
    remaining = total - p[idx]
    if remaining <= 0:
        p[idx] = 1.0
        return p
    scale = (1.0 - min_prob) / remaining
    for i in range(len(p)):
        if i != idx:
            p[i] *= scale
    p[idx] = min_prob
    s = p.sum()
    return p / s if s > 0 else p


def apply_rule_adjustment(
    proba: np.ndarray,
    class_list: np.ndarray,
    area_type: str,
    dt,
    facility_tag: Optional[str] = None,
    weather: Optional[dict] = None
) -> np.ndarray:
    """예측 확률(proba)에 시간/구역/시설/날씨 가중치 적용 후 정규화."""
    p = proba.copy()

    idx_cong = get_class_index(class_list, "혼잡")
    if idx_cong is None:
        return proba

    hour_f = dt.hour + dt.minute / 60.0
    is_weekend = dt.dayofweek in (5, 6)

    # 주말 기본 가중 (10~18 혼잡 ↑)
    if is_weekend:
        s, e = WEEKEND_RULE["window"]
        if _hhmm_to_float(s) <= hour_f < _hhmm_to_float(e):
            p[idx_cong] *= WEEKEND_RULE.get("factor_up", 1.0)

    # 구역 유형 가중
    rw = RULE_WEIGHTS.get(area_type, {})
    if in_any_window(hour_f, rw.get("boost", [])):
        p[idx_cong] *= rw.get("factor_up", 1.0)
    if in_any_window(hour_f, rw.get("suppress", [])):
        p[idx_cong] *= rw.get("factor_down", 1.0)

    # 시설 태그 보정
    if facility_tag == "CULTURE":
        s, e = CULTURE_WEEKEND_WINDOW
        if is_weekend and _hhmm_to_float(s) <= hour_f < _hhmm_to_float(e):
            p[idx_cong] *= CULTURE_WEEKEND_FACTOR_UP
            p = _enforce_min_prob_for_class(p, idx_cong, CULTURE_WEEKEND_MIN_PROB)
    elif facility_tag == "PARK":
        pass
    elif facility_tag == "OFFICE":
        pass

    # 날씨 보정
    if weather:
        temp = weather.get("temp", None)
        main = (weather.get("main") or "").lower()
        precip = bool(weather.get("is_precip", False))

        if precip or any(k in main for k in ["rain", "snow", "thunder"]):
            if facility_tag in (None, "CULTURE"):
                p[idx_cong] *= 1.12
            if facility_tag == "PARK":
                p[idx_cong] *= 0.92

        if isinstance(temp, (int, float)):
            if temp >= 30 or temp <= -2:
                if facility_tag in (None, "CULTURE", "OFFICE"):
                    p[idx_cong] *= 1.08
                if facility_tag == "PARK":
                    p[idx_cong] *= 0.95
            if 15 <= temp <= 25 and any(k in main for k in ["clear", "cloud"]):
                if facility_tag == "PARK":
                    p[idx_cong] *= 1.05

    s = p.sum()
    if s > 0:
        p /= s
    return p


def congestion_score_from_proba(proba: np.ndarray, class_list: np.ndarray) -> float:
    """0~1 스코어(높을수록 혼잡): P(혼잡)+0.5*P(보통)."""
    idx_h = get_class_index(class_list, "혼잡")
    idx_b = get_class_index(class_list, "보통")
    p_h = proba[idx_h] if idx_h is not None else 0.0
    p_b = proba[idx_b] if idx_b is not None else 0.0
    return float(max(0.0, min(1.0, p_h + 0.5 * p_b)))


# =========================
# 백오프 / 폴백
# =========================
def find_neighbors_with_backoff(df_lots: pd.DataFrame, lat: float, lon: float,
                                base_radius_km: float, max_radius_km: float = 10.0, top_k: int = 5):
    seq = [base_radius_km]
    for r in [1.0, 2.0, 5.0, max_radius_km]:
        if r > base_radius_km:
            seq.append(r)

    for r in seq:
        tmp = df_lots.copy()
        tmp["dist_km"] = tmp.apply(
            lambda row: haversine_km(lat, lon, float(row["lat"]), float(row["lng"])),
            axis=1
        )
        cand = tmp[tmp["dist_km"] <= r].sort_values("dist_km").head(top_k)
        if len(cand):
            return cand.reset_index(drop=True), r
    return pd.DataFrame(), max_radius_km


def global_fallback_predict(clf, le: LabelEncoder, xcols: List[str], area_type_map: Dict[str, str],
                            arrival_str: str) -> Tuple[str, float]:
    now = pd.to_datetime(arrival_str)
    areas = list(area_type_map.keys())
    if not areas:
        probs = np.ones(len(le.classes_)) / len(le.classes_)
        return le.inverse_transform([int(np.argmax(probs))])[0], float(np.max(probs))
    areas = areas[:3] if len(areas) >= 3 else areas

    def build_row(areaCd):
        base = {
            "hour": now.hour, "minute": now.minute,
            "dow": now.dayofweek, "is_weekend": int(now.dayofweek in (5, 6)),
            "temp": 0.0, "humidity": 0.0, "pm10": 0.0,
            "areaCd": areaCd, "prkType": "UNK",
        }
        row = pd.DataFrame([base])
        row_oh = pd.get_dummies(row, columns=["areaCd", "prkType"], drop_first=False)
        row_oh = row_oh.reindex(columns=xcols, fill_value=0.0)
        return row_oh

    probs = []
    for a in areas:
        Xrow = build_row(a)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xrow)[0]
        else:
            pred = clf.predict(Xrow)[0]
            proba = np.zeros(len(le.classes_), dtype=float); proba[int(pred)] = 1.0
        proba = apply_rule_adjustment(proba, le.classes_, area_type_map.get(a, "MIXED"), now)
        probs.append(proba)

    mean_proba = np.mean(probs, axis=0)
    cls_id = int(np.argmax(mean_proba))
    return le.inverse_transform([cls_id])[0], float(np.max(mean_proba))


# =========================
# 학습 (GPU 우선 → CPU 폴백)
# =========================
def train_classifier(X: pd.DataFrame, y: np.ndarray, use_gpu: bool = True):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if XGB_AVAILABLE:
        num_class = int(len(np.unique(y)))
        params = {
            "objective": "multi:softprob",
            "num_class": num_class,
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "predictor": "gpu_predictor" if use_gpu else "auto",
            **XGB_PARAMS,
        }
        booster = xgb.XGBClassifier(**params)
        try:
            booster.fit(Xtr, ytr)
            yhat = booster.predict(Xte)
            print("=== XGBoost Classifier Report ({} ) ===".format(
                "GPU" if use_gpu else "CPU"
            ))
            print(classification_report(yte, yhat, digits=4))
            print(f"Acc: {accuracy_score(yte, yhat):.4f} | F1(w): {f1_score(yte, yhat, average='weighted'):.4f}")
            return booster
        except Exception as e:
            print(f"[경고] XGBoost 학습 실패 → RandomForest 폴백 (사유: {e})")

    clf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    print("=== RandomForest Classifier Report (CPU) ===")
    print(classification_report(yte, yhat, digits=4))
    print(f"Acc: {accuracy_score(yte, yhat):.4f} | F1(w): {f1_score(yte, yhat, average='weighted'):.4f}")
    return clf


# =========================
# (추가) 주차장 이름 필터: '주차'/'parking' 포함 + 버스/승하차 등 제외
# =========================
def _is_parking_name(name: Optional[str]) -> bool:
    if not isinstance(name, str):
        return False
    n = name.lower()
    positive = ("주차", "주차장", "parking")
    negative = ("관광버스", "버스전용", "승하차", "이륜차", "허용 구간")
    if any(neg in n for neg in negative):
        return False
    return any(pos in n for pos in positive)


# =========================
# 커맨드: train / predict
# =========================
def cmd_train(args):
    os.makedirs(args.model_dir, exist_ok=True)

    df = load_and_concat(args.data)
    lots_meta = build_lots_meta(df)
    area_type_map = infer_area_type(df)

    X, y, xcols, le = prepare_features(df)
    clf = train_classifier(X, y, use_gpu=not args.no_gpu)

    # 산출물 저장
    joblib.dump(clf, os.path.join(args.model_dir, "model_cls.joblib"))
    with open(os.path.join(args.model_dir, "label_classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": list(le.classes_)}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.model_dir, "xcols.json"), "w", encoding="utf-8") as f:
        json.dump({"xcols": list(X.columns)}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.model_dir, "area_type_map.json"), "w", encoding="utf-8") as f:
        json.dump(area_type_map, f, ensure_ascii=False, indent=2)
    lots_meta.to_csv(os.path.join(args.model_dir, "lots_meta.csv"), index=False, encoding="utf-8-sig")

    print(f"\n[저장 완료] {args.model_dir}")
    print("- model_cls.joblib")
    print("- label_classes.json")
    print("- xcols.json")
    print("- area_type_map.json")
    print("- lots_meta.csv")


def cmd_predict(args):
    # 아티팩트 로드
    clf = joblib.load(os.path.join(args.model_dir, "model_cls.joblib"))
    with open(os.path.join(args.model_dir, "label_classes.json"), "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    le = LabelEncoder(); le.fit(classes)
    with open(os.path.join(args.model_dir, "xcols.json"), "r", encoding="utf-8") as f:
        xcols = json.load(f)["xcols"]
    with open(os.path.join(args.model_dir, "area_type_map.json"), "r", encoding="utf-8") as f:
        area_type_map = json.load(f)
    lots_meta = pd.read_csv(os.path.join(args.model_dir, "lots_meta.csv"), encoding="utf-8")

    now = pd.to_datetime(args.arrival)

    # 1) 내 CSV 메타에서 후보 선택
    if args.exact_radius:
        tmp = lots_meta.copy()
        tmp["dist_km"] = tmp.apply(
            lambda row: haversine_km(args.lat, args.lon, float(row["lat"]), float(row["lng"])),
            axis=1
        )
        neighbors = tmp[tmp["dist_km"] <= args.radius].sort_values("dist_km").head(args.top_k).reset_index(drop=True)
        used_r = args.radius
    else:
        neighbors, used_r = find_neighbors_with_backoff(
            lots_meta, args.lat, args.lon, base_radius_km=args.radius, max_radius_km=10.0, top_k=args.top_k
        )

    # 2) 후보가 모자라면 Google Places로 보강(옵션)
    if (len(neighbors) < args.top_k) and args.fill_external and args.use_places:
        gkey = _read_key("GOOGLE_API_KEY", args.google_api_key)
        if gkey:
            ext_df = discover_parking_via_places(args.lat, args.lon, args.places_radius_m, gkey, args.top_k)
            if len(ext_df):
                base_cols = ["prkCd", "prkNm", "areaCd", "prkType", "lat", "lng", "dist_km"]
                neighbors = neighbors.copy()
                if "dist_km" not in neighbors.columns:
                    neighbors["dist_km"] = neighbors.apply(
                        lambda row: haversine_km(args.lat, args.lon, float(row["lat"]), float(row["lng"])),
                        axis=1
                    )
                combo = pd.concat(
                    [neighbors[base_cols] if len(neighbors) else neighbors, ext_df[base_cols + ["types"]]],
                    ignore_index=True
                )
                combo = combo.drop_duplicates(subset=["prkCd"]).sort_values("dist_km").head(args.top_k).reset_index(drop=True)
                neighbors = combo

    # 후보가 전혀 없으면 폴백
    if len(neighbors) == 0:
        label, conf = global_fallback_predict(clf, le, xcols, area_type_map, args.arrival)
        # JSON 먼저 쓰기
        if args.out_json:
            try:
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump({
                        "meta": {"arrival": args.arrival, "used_radius_km": float(used_r), "top_k": int(args.top_k)},
                        "items": [],
                        "alternatives": [],
                        "summary": {"pooled_level": label, "confidence": round(conf, 4)}
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        print(f"[예측 시각] {args.arrival}")
        print(f"[검색 반경] {used_r:.1f} km (주변 주차장 없음 → 폴백)")
        print(f"[혼잡도 등급] {label} (신뢰도≈{conf:.2f})")
        return

    # (선택) 주변 카테고리/날씨 컨텍스트
    facility_ctx_tag = None
    weather_ctx = None
    if args.use_places:
        gkey = _read_key("GOOGLE_API_KEY", args.google_api_key)
        if gkey:
            types = fetch_place_types_google(args.lat, args.lon, int(max(200, args.radius * 1000)), gkey, args.model_dir)
            facility_ctx_tag = map_types_to_facility_tag(types) or facility_ctx_tag
    if args.use_weather:
        wkey = _read_key("OWM_API_KEY", args.owm_api_key)
        if wkey:
            weather_ctx = fetch_weather_owm(args.lat, args.lon, now, wkey, args.model_dir)

    # 개별 주차장 예측 준비
    def build_row(areaCd: str, prkType: str):
        base = {
            "hour": now.hour, "minute": now.minute,
            "dow": now.dayofweek, "is_weekend": int(now.dayofweek in (5, 6)),
            "temp": 0.0, "humidity": 0.0, "pm10": 0.0,
            "areaCd": areaCd, "prkType": prkType if isinstance(prkType, str) and prkType else "UNK",
        }
        row = pd.DataFrame([base])
        row_oh = pd.get_dummies(row, columns=["areaCd", "prkType"], drop_first=False)
        row_oh = row_oh.reindex(columns=xcols, fill_value=0.0)
        return row_oh

    rows = []
    for _, lot in neighbors.iterrows():
        Xrow = build_row(lot.get("areaCd", "EXT"), lot.get("prkType", "EXT"))
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xrow)[0]
        else:
            pred = clf.predict(Xrow)[0]
            proba = np.zeros(len(le.classes_), dtype=float); proba[int(pred)] = 1.0

        # lot별 시설 태그(있으면 우선), 없으면 컨텍스트 태그
        lot_facility_tag = None
        if "types" in neighbors.columns:
            lot_facility_tag = map_types_to_facility_tag(lot.get("types", []))
        area_t = area_type_map.get(lot.get("areaCd", "EXT"), "MIXED")
        if lot_facility_tag == "CULTURE":
            area_t = "ENTERTAINMENT"
        elif lot_facility_tag == "OFFICE":
            area_t = "OFFICE"
        elif lot_facility_tag == "PARK":
            area_t = "MIXED"

        proba_adj = apply_rule_adjustment(
            proba, le.classes_, area_t, now,
            facility_tag=lot_facility_tag or facility_ctx_tag,
            weather=weather_ctx
        )

        idx_max = int(np.argmax(proba_adj))
        level_txt = le.inverse_transform([idx_max])[0]
        score = congestion_score_from_proba(proba_adj, le.classes_)

        def _p(cls_name):
            idx = get_class_index(le.classes_, cls_name)
            return round(float(proba_adj[idx]), 4) if idx is not None else 0.0

        # (변경) lat/lng 추가(콘솔 출력은 기존 prkCd/areaCd 사용하므로 그대로 보존)
        rows.append({
            "prkCd": lot.get("prkCd"),
            "prkNm": lot.get("prkNm"),
            "areaCd": lot.get("areaCd"),
            "lat": float(lot.get("lat", 0.0)),
            "lng": float(lot.get("lng", 0.0)),
            "dist_km": float(lot.get("dist_km", 0.0)),
            "pred_level": level_txt,
            "score": round(float(score), 4),
            "p_여유": _p("여유"),
            "p_보통": _p("보통"),
            "p_혼잡": _p("혼잡"),
        })

    df_out = pd.DataFrame(rows)

    # 정렬
    if args.sort_by == "score":
        df_out = df_out.sort_values(["score", "dist_km"], ascending=[False, True]).reset_index(drop=True)
    else:
        df_out = df_out.sort_values(["dist_km", "score"], ascending=[True, False]).reset_index(drop=True)

    # 대표 등급(리스트의 1위)
    representative_level = str(df_out.iloc[0]["pred_level"]) if len(df_out) else None
    is_congested_now = (representative_level == "혼잡")

    # --- 대체 추천 계산 ---
    alt_list: List[dict] = []
    if is_congested_now:
        alt_mode = getattr(args, "alt_mode", "ring")
        alt_max_km = float(getattr(args, "alt_max_km", 3.0))
        alt_top_k = int(getattr(args, "alt_top_k", 5))

        if alt_mode == "recenter":
            alt_center_lat = float(args.alt_center_lat)
            alt_center_lon = float(args.alt_center_lon)
            alt_base = lots_meta.copy()
            alt_base["alt_dist_km"] = alt_base.apply(
                lambda row: haversine_km(alt_center_lat, alt_center_lon, float(row["lat"]), float(row["lng"])),
                axis=1
            )
            alt_cands = alt_base[alt_base["alt_dist_km"] <= alt_max_km].copy()
        else:  # ring
            base = lots_meta.copy()
            base["dist_km"] = base.apply(
                lambda row: haversine_km(args.lat, args.lon, float(row["lat"]), float(row["lng"])),
                axis=1
            )
            alt_cands = base[(base["dist_km"] > used_r) & (base["dist_km"] <= alt_max_km)].copy()

        # (추가) 대체 후보는 '주차장'만 남기기
        if len(alt_cands):
            alt_cands = alt_cands[alt_cands["prkNm"].apply(_is_parking_name)].copy()

        if len(alt_cands):
            alt_rows = []
            for _, lot in alt_cands.iterrows():
                Xrow = build_row(lot.get("areaCd", "EXT"), lot.get("prkType", "EXT"))
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(Xrow)[0]
                else:
                    pred = clf.predict(Xrow)[0]
                    proba = np.zeros(len(le.classes_), dtype=float); proba[int(pred)] = 1.0

                area_t = area_type_map.get(lot.get("areaCd", "EXT"), "MIXED")
                proba_adj = apply_rule_adjustment(
                    proba, le.classes_, area_t, now,
                    facility_tag=facility_ctx_tag,
                    weather=weather_ctx
                )

                def _p(cls_name):
                    idx = get_class_index(le.classes_, cls_name)
                    return round(float(proba_adj[idx]), 4) if idx is not None else 0.0

                idx_max = int(np.argmax(proba_adj))
                level_txt = le.inverse_transform([idx_max])[0]
                dist_val = float(lot["alt_dist_km"] if (alt_mode == "recenter") else lot["dist_km"])
                alt_rows.append({
                    "prkCd": lot.get("prkCd"),
                    "prkNm": lot.get("prkNm"),
                    "areaCd": lot.get("areaCd"),
                    "lat": float(lot.get("lat", 0.0)),
                    "lng": float(lot.get("lng", 0.0)),
                    "dist_km": dist_val,
                    "pred_level": level_txt,
                    "p_여유": _p("여유"),
                    "p_보통": _p("보통"),
                    "p_혼잡": _p("혼잡"),
                    "score": float(_p("혼잡") + 0.5 * _p("보통")),
                })

            alt_df = pd.DataFrame(alt_rows)
            alt_df = alt_df[alt_df["pred_level"].isin(["여유", "보통"])].copy()
            if len(alt_df):
                alt_df = alt_df.sort_values(by=["score", "dist_km"], ascending=[False, True]) \
                               .head(alt_top_k).reset_index(drop=True)
                alt_list = alt_df.to_dict(orient="records")

    # ===== JSON은 무조건 먼저 기록 =====
    if args.out_json:
        try:
            # (추가) 반환 필드 정리: areaCd/prkCd 제외, lat/lon만 노출
            items_slim = []
            for r in df_out.head(args.top_k).to_dict(orient="records"):
                items_slim.append({
                    "name": r.get("prkNm"),
                    "lat": float(r.get("lat", 0.0)),
                    "lon": float(r.get("lng", 0.0)),
                    "dist_km": float(r.get("dist_km", 0.0)),
                    "pred_level": r.get("pred_level"),
                    "score": float(r.get("score", 0.0)),
                    "p_relaxed": float(r.get("p_여유", 0.0)),
                    "p_normal": float(r.get("p_보통", 0.0)),
                    "p_congested": float(r.get("p_혼잡", 0.0)),
                })

            alts_slim = []
            for r in alt_list:
                alts_slim.append({
                    "name": r.get("prkNm"),
                    "lat": float(r.get("lat", 0.0)),
                    "lon": float(r.get("lng", 0.0)),
                    "dist_km": float(r.get("dist_km", 0.0)),
                    "pred_level": r.get("pred_level"),
                    "score": float(r.get("score", 0.0)),
                    "p_relaxed": float(r.get("p_여유", 0.0)),
                    "p_normal": float(r.get("p_보통", 0.0)),
                    "p_congested": float(r.get("p_혼잡", 0.0)),
                })

            result = {
                "meta": {
                    "arrival": args.arrival,
                    "used_radius_km": float(used_r),
                    "top_k": int(args.top_k),
                    "sort_by": args.sort_by,
                    "list_mode": bool(args.list_mode),
                    "representative_level": representative_level,
                    "alt_mode": getattr(args, "alt_mode", "ring"),
                    "alt_max_km": float(getattr(args, "alt_max_km", 3.0)),
                },
                "items": items_slim,
                "alternatives": alts_slim,
            }
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            try:
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump({"error": f"failed to write json: {e.__class__.__name__}: {e}"}, f, ensure_ascii=False)
            except Exception:
                pass

    # ===== 콘솔 출력 =====
    if args.list_mode:
        print(f"[예측 시각] {args.arrival}")
        print(f"[검색 반경] {used_r:.1f} km")
        print(f"[후보 수] {len(df_out)} (요청 top_k={args.top_k})")
        preview = df_out.head(args.top_k)
        for i, r in preview.iterrows():
            print(f"{i+1:02d}. [{r['pred_level']}] {r['prkNm']} (코드:{r['prkCd']}, 구역:{r['areaCd']}) "
                  f"거리 {r['dist_km']:.2f}km | 점수 {r['score']:.2f} "
                  f"(여:{r['p_여유']:.2f}/보:{r['p_보통']:.2f}/혼:{r['p_혼잡']:.2f})")
        if is_congested_now and len(alt_list):
            title = "[대체 추천: {}] {}".format(
                "새 좌표 중심" if getattr(args, "alt_mode", "ring") == "recenter" else "초기 반경 밖",
                "반경 {:.1f}km 이내".format(float(getattr(args, "alt_max_km", 3.0)))
                if getattr(args, "alt_mode", "ring") == "recenter"
                else "구간 ({:.1f}~{:.1f}] km".format(float(used_r), float(getattr(args, "alt_max_km", 3.0)))
            )
            print("\n" + title)
            for r in alt_list:
                print(f"- {r['prkNm']} (코드:{r['prkCd']}, 구역:{r['areaCd']}) "
                      f"거리 {r['dist_km']:.2f}km | 예측:{r['pred_level']} "
                      f"(여:{r['p_여유']:.2f}/보:{r['p_보통']:.2f}/혼:{r['p_혼잡']:.2f})")
    else:
        # 리스트 모드 OFF: 거리 가중 풀링 후 단일 등급
        weights = 1.0 / df_out["dist_km"].clip(lower=1e-3)
        denom = weights.sum()
        p_y = (df_out["p_여유"] * weights).sum() / denom
        p_b = (df_out["p_보통"] * weights).sum() / denom
        p_h = (df_out["p_혼잡"] * weights).sum() / denom
        pooled_vec = np.array([p_y, p_b, p_h], dtype=float)
        proba_pooled = np.zeros(len(le.classes_), dtype=float)
        for prob, cname in zip(pooled_vec, ["여유", "보통", "혼잡"]):
            idx = get_class_index(le.classes_, cname)
            if idx is not None:
                proba_pooled[idx] = prob
        cls_id = int(np.argmax(proba_pooled))
        cls_txt = le.inverse_transform([cls_id])[0]
        conf = float(np.max(proba_pooled))
        print(f"[예측 시각] {args.arrival}")
        print(f"[검색 반경] {used_r:.1f} km")
        print(f"[혼잡도 등급] {cls_txt} (신뢰도≈{conf:.2f})")


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="모델 학습")
    tr.add_argument("--data", required=True, help="'../data/*.csv' 또는 'C:/.../data/*.csv'")
    tr.add_argument("--model_dir", required=True, help="모델/아티팩트 저장 폴더")
    tr.add_argument("--no_gpu", action="store_true", help="강제로 CPU 사용(XGBoost GPU 비활성)")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="예측")
    pr.add_argument("--model_dir", required=True, help="모델/아티팩트 폴더")
    pr.add_argument("--lat", type=float, required=True, help="위도")
    pr.add_argument("--lon", type=float, required=True, help="경도")
    pr.add_argument("--arrival", required=True, help="도착 예정 시각 'YYYY-MM-DDTHH:MM' 또는 'YYYY-MM-DD HH:MM'")
    pr.add_argument("--radius", type=float, default=0.5, help="초기 반경(km)")
    pr.add_argument("--top_k", type=int, default=15, help="반환할 최대 후보 수")

    # 리스트 모드 & 저장 옵션
    pr.add_argument("--list_mode", action="store_true", help="반경 내 주차장 리스트(개별 예측) 출력")
    pr.add_argument("--exact_radius", action="store_true", help="반경 고정(백오프 비활성, 예: 정확히 1km)")
    pr.add_argument("--sort_by", default="score", choices=["score", "distance"], help="정렬 기준")
    pr.add_argument("--out_csv", default=None, help="결과 CSV 저장 경로")
    pr.add_argument("--out_json", default=None, help="결과 JSON 저장 경로")

    # 외부 주차장 보강 / API 키
    pr.add_argument("--fill_external", action="store_true", help="CSV 후보가 부족하면 Google Places로 보강")
    pr.add_argument("--places_radius_m", type=int, default=1000, help="Places 검색 반경(m)")
    pr.add_argument("--use_places", action="store_true", help="주변 카테고리/외부 후보 탐색을 위해 Places 사용")
    pr.add_argument("--use_weather", action="store_true", help="OpenWeather OneCall 사용")
    pr.add_argument("--google_api_key", default=None, help="GOOGLE_API_KEY 환경변수 사용")
    pr.add_argument("--owm_api_key", default=None, help="OWM_API_KEY 환경변수 사용")

    # --- 대체 추천 옵션 ---
    pr.add_argument("--alt_mode", default="ring", choices=["ring", "recenter"],
                    help="대체 추천 검색 모드: 'ring'(초기 반경 밖 링) 또는 'recenter'(새 좌표 중심)")
    pr.add_argument("--alt_max_km", type=float, default=3.0,
                    help="대체 추천 최대 검색 반경(km). ring 모드에선 (radius, alt_max_km] 구간을 사용")
    pr.add_argument("--alt_top_k", type=int, default=5,
                    help="대체 추천 최대 개수")
    pr.add_argument("--alt_center_lat", type=float, default=None,
                    help="recenter 모드에서 사용할 중심 위도")
    pr.add_argument("--alt_center_lon", type=float, default=None,
                    help="recenter 모드에서 사용할 중심 경도")

    pr.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    # recenter 모드 필수 파라미터 체크
    if args.cmd == "predict" and args.alt_mode == "recenter":
        if args.alt_center_lat is None or args.alt_center_lon is None:
            raise SystemExit("recenter 모드에는 --alt_center_lat, --alt_center_lon 이 필요합니다.")
    args.func(args)


if __name__ == "__main__":
    main()
