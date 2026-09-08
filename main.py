import os
import math
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import s3fs
import xarray as xr
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

API_KEY = os.environ.get("LIGHTNING_INGESTOR_KEY", "").strip()
ALLOW_INSECURE_DEV = os.environ.get("ALLOW_INSECURE_DEV", "false").lower() == "true"
BUCKET = os.environ.get("GOES_BUCKET", "noaa-goes19")
PRODUCT = os.environ.get("GOES_PRODUCT", "GLM-L2-LCFA")
MAX_LOOKBACK_MINUTES = int(os.environ.get("MAX_LOOKBACK_MINUTES", "120"))
MAX_BBOX_SPAN_DEG = float(os.environ.get("MAX_BBOX_SPAN_DEG", "20"))
LATEST_GRACE_SECONDS = int(os.environ.get("LATEST_GRACE_SECONDS", "120"))
MAX_COVERAGE_GAP_SECONDS = int(os.environ.get("MAX_COVERAGE_GAP_SECONDS", "45"))

app = FastAPI(title="HeatSafe Oklahoma Lightning Ingestor", version="1.0.0")
_fs = s3fs.S3FileSystem(anon=True)
_fs_lock = threading.Lock()


def utcnow():
    return datetime.now(timezone.utc)


def check_key(x_api_key: str = Header(default="")):
    if not API_KEY and not ALLOW_INSECURE_DEV:
        raise HTTPException(status_code=503, detail="ingestor API key is not configured")
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_bbox(bbox: str):
    try:
        min_lat, min_lon, max_lat, max_lon = [float(x.strip()) for x in bbox.split(",")]
    except Exception:
        raise HTTPException(status_code=400, detail="bbox must be minLat,minLon,maxLat,maxLon")
    vals = [min_lat, min_lon, max_lat, max_lon]
    if not all(math.isfinite(v) for v in vals):
        raise HTTPException(status_code=400, detail="bbox contains non-finite coordinates")
    if not (-90 <= min_lat < max_lat <= 90 and -180 <= min_lon < max_lon <= 180):
        raise HTTPException(status_code=400, detail="bbox coordinates are invalid")
    if (max_lat - min_lat) > MAX_BBOX_SPAN_DEG or (max_lon - min_lon) > MAX_BBOX_SPAN_DEG:
        raise HTTPException(status_code=400, detail="bbox is too large")
    return min_lat, min_lon, max_lat, max_lon


def parse_filename_window(fname: str):
    base = os.path.basename(fname)
    try:
        s_token = base.split("_s", 1)[1].split("_e", 1)[0]
        e_token = base.split("_e", 1)[1].split("_c", 1)[0]
        return _goes_token_to_dt(s_token), _goes_token_to_dt(e_token)
    except Exception:
        return None, None


def _goes_token_to_dt(token: str):
    core = token[:13]
    year = int(core[:4]); doy = int(core[4:7]); hh = int(core[7:9]); mm = int(core[9:11]); ss = int(core[11:13])
    frac = token[13:]
    micro = 0
    if frac and frac.isdigit():
        micro = int(float("0." + frac) * 1_000_000)
    return datetime(year, 1, 1, hh, mm, ss, microsecond=micro, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def _hour_prefix(dt: datetime):
    return f"{BUCKET}/{PRODUCT}/{dt.strftime('%Y')}/{dt.strftime('%j')}/{dt.strftime('%H')}/"


def list_glm_files(since_dt: datetime, now_dt: datetime):
    files = []
    cur = since_dt.replace(minute=0, second=0, microsecond=0)
    last_hour = now_dt.replace(minute=0, second=0, microsecond=0)
    while cur <= last_hour:
        try:
            with _fs_lock:
                files.extend(_fs.ls(_hour_prefix(cur)))
        except Exception:
            pass
        cur += timedelta(hours=1)

    selected = []
    for path in files:
        start, end = parse_filename_window(path)
        if not start or not end:
            continue
        if end >= since_dt - timedelta(seconds=30) and start <= now_dt + timedelta(seconds=30):
            selected.append((path, start, end))
    selected.sort(key=lambda x: x[1])
    return selected


def coverage_for_files(files, since_dt, now_dt):
    if not files:
        return False, int((now_dt - since_dt).total_seconds()), None, None
    expected_end = now_dt - timedelta(seconds=LATEST_GRACE_SECONDS)
    cursor = since_dt
    total_gap = 0.0
    first_start = files[0][1]
    last_end = files[-1][2]
    for _, start, end in files:
        if end < since_dt:
            continue
        if start > cursor:
            total_gap += (start - cursor).total_seconds()
        if end > cursor:
            cursor = end
    if cursor < expected_end:
        total_gap += (expected_end - cursor).total_seconds()
    coverage_complete = total_gap <= MAX_COVERAGE_GAP_SECONDS and last_end >= expected_end
    return coverage_complete, max(0, int(round(total_gap))), first_start, last_end


def _to_scalar(v):
    try:
        return v.item()
    except Exception:
        return v


def decode_flash_time(value, file_start: datetime):
    v = _to_scalar(value)
    if isinstance(v, np.datetime64):
        ns = v.astype("datetime64[ns]").astype(np.int64)
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    if isinstance(v, np.timedelta64):
        seconds = v / np.timedelta64(1, "s")
        return file_start + timedelta(seconds=float(seconds))
    try:
        seconds = float(v)
        if math.isfinite(seconds):
            return file_start + timedelta(seconds=seconds)
    except Exception:
        pass
    return file_start


def _read_var(ds, *names):
    for name in names:
        if name in ds.variables:
            return np.asarray(ds[name].values)
    return None


def read_flashes(path: str, file_start: datetime, bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    with _fs_lock:
        file_obj = _fs.open(path, "rb")
    ds = xr.open_dataset(file_obj, engine="h5netcdf", decode_times=True, mask_and_scale=True)
    try:
        lat = _read_var(ds, "flash_lat")
        lon = _read_var(ds, "flash_lon")
        first_time = _read_var(ds, "flash_time_offset_of_first_event", "flash_time_offset", "flash_time")
        last_time = _read_var(ds, "flash_time_offset_of_last_event")
        energy = _read_var(ds, "flash_energy")
        area = _read_var(ds, "flash_area")
        flash_id = _read_var(ds, "flash_id")
        dqf = _read_var(ds, "flash_quality_flag", "flash_DQF")
        if lat is None or lon is None or first_time is None or flash_id is None:
            raise ValueError("required GLM flash variables are missing")

        out = []
        file_key = os.path.basename(path).split("_c", 1)[0]
        n = min(len(lat), len(lon), len(first_time), len(flash_id))
        for i in range(n):
            try:
                la = float(lat[i]); lo = float(lon[i])
            except Exception:
                continue
            if not math.isfinite(la) or not math.isfinite(lo):
                continue
            if la < min_lat or la > max_lat or lo < min_lon or lo > max_lon:
                continue
            if dqf is not None:
                try:
                    if int(_to_scalar(dqf[i])) != 0:
                        continue
                except Exception:
                    pass
            event_dt = decode_flash_time(first_time[i], file_start)
            end_dt = decode_flash_time(last_time[i], file_start) if last_time is not None else event_dt
            try:
                raw_id = int(_to_scalar(flash_id[i]))
            except Exception:
                raw_id = i
            stable_id = f"{file_key}:{raw_id}"

            def finite_or_none(arr):
                if arr is None:
                    return None
                try:
                    x = float(_to_scalar(arr[i]))
                    return x if math.isfinite(x) else None
                except Exception:
                    return None

            out.append({
                "id": stable_id,
                "lat": la,
                "lon": lo,
                "time": event_dt.isoformat().replace("+00:00", "Z"),
                "end_time": end_dt.isoformat().replace("+00:00", "Z"),
                "energy": finite_or_none(energy),
                "area": finite_or_none(area),
            })
        return out
    finally:
        ds.close()
        try:
            file_obj.close()
        except Exception:
            pass


@app.get("/lightning")
def lightning(bbox: str = Query(...), since: str = Query(...), source: str = Query("goes19"), x_api_key: str = Header(default="")):
    check_key(x_api_key)
    if source not in ("goes19", "goes19_glm_l2_lcfa"):
        raise HTTPException(status_code=400, detail="unsupported source")
    try:
        since_dt = parse_iso(since)
    except Exception:
        return JSONResponse({"ok": False, "status": "error", "error": "invalid since"}, status_code=400)
    now_dt = utcnow()
    max_lookback = timedelta(minutes=MAX_LOOKBACK_MINUTES)
    if since_dt > now_dt + timedelta(minutes=1):
        return JSONResponse({"ok": False, "status": "error", "error": "since is in the future"}, status_code=400)
    if now_dt - since_dt > max_lookback:
        since_dt = now_dt - max_lookback

    parsed_bbox = parse_bbox(bbox)
    files = list_glm_files(since_dt, now_dt)
    coverage_complete, coverage_gap_seconds, actual_start, actual_end = coverage_for_files(files, since_dt, now_dt)

    flashes = []
    file_errors = []
    for path, start, _end in files:
        try:
            flashes.extend(read_flashes(path, start, parsed_bbox))
        except Exception as exc:
            file_errors.append({"file": os.path.basename(path), "error": str(exc)[:240]})

    if file_errors:
        coverage_complete = False
        coverage_gap_seconds = max(coverage_gap_seconds, 20 * len(file_errors))

    dedup = {fl["id"]: fl for fl in flashes}
    uniq = sorted(dedup.values(), key=lambda f: f["time"])
    status = "ok" if coverage_complete else "degraded"
    return {
        "ok": True,
        "source": "goes19_glm_l2_lcfa",
        "status": status,
        "generated_at": utcnow().isoformat().replace("+00:00", "Z"),
        "requested_since": since,
        "actual_data_window_start": (actual_start or since_dt).isoformat().replace("+00:00", "Z"),
        "actual_data_window_end": (actual_end or since_dt).isoformat().replace("+00:00", "Z"),
        "coverage_complete": bool(coverage_complete),
        "coverage_gap_seconds": int(coverage_gap_seconds),
        "file_count": len(files),
        "file_error_count": len(file_errors),
        "flash_count": len(uniq),
        "flashes": uniq,
    }


@app.get("/health")
def health():
    now_dt = utcnow()
    files = list_glm_files(now_dt - timedelta(minutes=5), now_dt)
    latest_end = files[-1][2] if files else None
    age = (now_dt - latest_end).total_seconds() if latest_end else None
    healthy_feed = latest_end is not None and age <= LATEST_GRACE_SECONDS + 60
    return {
        "ok": True,
        "service": "heatsafe-lightning-ingestor",
        "source": "goes19_glm_l2_lcfa",
        "api_key_configured": bool(API_KEY),
        "feed_reachable": bool(files),
        "feed_fresh": bool(healthy_feed),
        "latest_product_end": latest_end.isoformat().replace("+00:00", "Z") if latest_end else None,
        "latest_age_seconds": round(age, 1) if age is not None else None,
        "checked_at": now_dt.isoformat().replace("+00:00", "Z"),
    }