def normalize_label(lbl: str) -> str:
    return lbl.strip().upper().rstrip(".")

def ensure_schema(result: dict) -> dict:
    # normalize label
    result["label"] = normalize_label(result.get("label","CANNOT_RECOGNIZE"))
    # ensure keywords list
    kws = result.get("keywords") or []
    if not isinstance(kws, list):
        kws = [k.strip() for k in kws.split(",") if k.strip()]
    result["keywords"] = kws
    return result
