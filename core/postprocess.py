#core/postprocess.py
def normalize_label(lbl: str) -> str:
    return lbl.strip().upper().rstrip(".")

def ensure_schema(result: dict) -> dict:
    result["label"] = normalize_label(result.get("label", "CANNOT_RECOGNIZE"))
    kws = result.get("keywords") or []
    if not isinstance(kws, list):
        kws = [k.strip() for k in kws.split(",") if k.strip()]
    result["keywords"] = kws
    result["confidence_score"] = round(float(result.get("confidence", 0.5)), 2)
    result["time_taken_secs"] = float(result.get("time_taken_secs", 0.0))
    return result
