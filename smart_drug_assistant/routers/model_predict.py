import os
import io
import re
from typing import Any
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from ultralytics import YOLO
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TrOCRProcessor

# --- Router ---
router = APIRouter()

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HELPER: model directory ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOAD YOLO MODEL ONCE ---
SHAPE_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
if not os.path.exists(SHAPE_MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at {SHAPE_MODEL_PATH}")
shape_model = YOLO(SHAPE_MODEL_PATH)

# --- LOAD MULTITASK PILL MODEL ONCE ---
DEPLOY_PATH = os.path.join(MODEL_DIR, "pill_multitask_deploy.pt")
if not os.path.exists(DEPLOY_PATH):
    raise FileNotFoundError(f"Multitask model checkpoint not found at {DEPLOY_PATH}")
deploy_checkpoint = torch.load(DEPLOY_PATH, map_location=device)

from .pill_model import PillMultiTaskModel


# Recreate MultiLabelBinarizer
mlb_classes = deploy_checkpoint["mlb_classes"]
mlb = MultiLabelBinarizer(classes=mlb_classes)
mlb.fit([mlb_classes])

# Recreate model
base_colors = deploy_checkpoint["base_colors"]
MODEL_NAME = deploy_checkpoint["model_name"]
deploy_model = PillMultiTaskModel(model_name=MODEL_NAME, num_labels=len(base_colors)).to(device)
deploy_model.load_state_dict(deploy_checkpoint["model_state"])
deploy_model.eval()

# Load OCR processor
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

# --- HELPER FUNCTION ---
def _normalize_imprint(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _tokenize_db_imprint(s: str) -> list[str]:
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", (s or "")) if t]
    return [_normalize_imprint(t) for t in tokens if _normalize_imprint(t)]


def _tokenize_predicted_imprint(s: str) -> list[str]:
    # Same tokenization as DB; drop extremely short tokens to reduce noise.
    toks = _tokenize_db_imprint(s)
    return [t for t in toks if len(t) >= 2]


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    a = _normalize_imprint(a)
    b = _normalize_imprint(b)
    if not a or not b:
        return 0.0
    dist = _edit_distance(a, b)
    denom = max(len(a), len(b))
    return max(0.0, 1.0 - (dist / denom))


def _confusable_regex_token(token: str) -> str:
    token = _normalize_imprint(token)
    if not token:
        return ""

    conf = {
        "0": "[0O]",
        "O": "[0O]",
        "1": "[1IL]",
        "I": "[1IL]",
        "L": "[1IL]",
        "2": "[2Z]",
        "Z": "[2Z]",
        "5": "[5S]",
        "S": "[5S]",
        "8": "[8B]",
        "B": "[8B]",
        "M": "[MW]",
        "W": "[MW]",
    }

    parts = []
    for ch in token:
        parts.append(conf.get(ch, re.escape(ch)))
    return "".join(parts)


def _choose_color_filters(color_scores: dict[str, float]) -> list[str]:
    items = sorted(color_scores.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return []

    # Always keep top-1. Add more colors only if the model is confident enough.
    # This avoids filtering on a random second color (e.g., 0.01) which kills recall.
    (c1, p1) = items[0]
    chosen = [c1]

    if len(items) >= 2:
        (c2, p2) = items[1]
        if p2 >= 0.25 or abs(p1 - p2) < 0.10:
            chosen.append(c2)

    # Only consider a third color if it is really close to the second, and reasonably high.
    if len(items) >= 3 and len(chosen) == 2:
        (c3, p3) = items[2]
        if p3 >= 0.25 and abs(p2 - p3) < 0.05:
            chosen.append(c3)

    return chosen


def _clean_ocr_imprint(raw: str) -> str:
    # Keep imprint short and conservative to reduce hallucinated long strings.
    # DB imprints tend to look like: "WATSON;384", "P;300", "GG;575", "IP;101".
    s = (raw or "").upper()
    s = s.replace(" ", ";")
    s = re.sub(r"[^A-Z0-9;]", "", s)
    s = re.sub(r";{2,}", ";", s).strip(";")
    sanitized = s

    def canonicalize(tokens_in: list[str]) -> str:
        # Canonical DB-like format: TOKEN(;TOKEN)* where tokens are mostly A-Z and 2-4 digits.
        tokens: list[str] = []
        for t in tokens_in:
            t = _normalize_imprint(t)
            if not t:
                continue

            # Extract alpha and digit chunks even from mixed runs.
            # Examples:
            # - P3002 -> P ; 300
            # - AN384W384384 -> AN ; 384 ; W ; 384
            for m in re.finditer(r"[A-Z]{1,8}", t):
                tokens.append(m.group(0))
            for m in re.finditer(r"[0-9]{2,6}", t):
                digits = m.group(0)
                # Keep a 4-digit candidate if present, but also keep a 3-digit candidate
                # for cases where OCR hallucinates an extra trailing digit (e.g., 3843 -> 384).
                if len(digits) >= 4:
                    tokens.append(digits[:3])
                    tokens.append(digits[:4])
                else:
                    tokens.append(digits)

        # Prefer longest alpha token + best numeric token if both exist.
        alpha = [t for t in tokens if re.fullmatch(r"[A-Z]{1,8}", t)]
        nums3 = [t for t in tokens if re.fullmatch(r"[0-9]{3}", t)]
        nums4 = [t for t in tokens if re.fullmatch(r"[0-9]{4}", t)]
        nums2 = [t for t in tokens if re.fullmatch(r"[0-9]{2}", t)]
        alpha_best = max(alpha, key=len) if alpha else ""

        # Heuristic aligned with dataset:
        # - Manufacturer-like alpha (len>=3) usually pairs with a 3-digit code (e.g., WATSON;384)
        # - Two-letter alpha often pairs with a 4-digit code (e.g., TV;1003)
        # - Single-letter alpha can be either 3-digit or 4-digit; prefer 4-digit when available
        #   (e.g., V;3566), otherwise fall back to 3-digit (e.g., P;300)
        num_best = ""
        if alpha_best and len(alpha_best) >= 3:
            num_best = (nums3[0] if nums3 else (nums4[0] if nums4 else (nums2[0] if nums2 else "")))
        elif alpha_best and len(alpha_best) == 1:
            num_best = (nums4[0] if nums4 else (nums3[0] if nums3 else (nums2[0] if nums2 else "")))
        else:
            num_best = (nums4[0] if nums4 else (nums3[0] if nums3 else (nums2[0] if nums2 else "")))

        out: list[str] = []
        if alpha_best:
            out.append(alpha_best)
        if num_best:
            out.append(num_best)

        # Fallback: if we only have one type, keep up to 2 tokens.
        if not out:
            out = tokens[:2]
        return ";".join(out)

    # Extract tokens and canonicalize.
    raw_tokens = [t for t in re.split(r"[^A-Za-z0-9]+", s) if t]
    s = canonicalize(raw_tokens)

    # Never return empty: fall back to sanitized OCR (truncated) if canonicalization fails.
    if not s:
        s = sanitized

    # Final length guard
    if len(s) > 24:
        s = s[:24].rstrip(";")
    return s


def predict_multitask(image: Image.Image):
    """Predict pill imprint and colors using the multitask model."""
    deploy_model.eval()
    image_tensor = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        # OCR Prediction
        gen_model = deploy_model.module.trocr if hasattr(deploy_model, "module") else deploy_model.trocr
        generated_ids = gen_model.generate(
            image_tensor,
            max_length=12,
            num_beams=3,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15
        )
        predicted_text_raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predicted_text = _clean_ocr_imprint(predicted_text_raw)

        # Color Prediction
        _, color_logits = deploy_model(image_tensor)
        color_probs = torch.sigmoid(color_logits).squeeze(0).detach().cpu()

        color_scores = {c: float(color_probs[i].item()) for i, c in enumerate(base_colors)}
        predicted_colors = [c for c, p in sorted(color_scores.items(), key=lambda kv: kv[1], reverse=True) if p > 0.5]

    return predicted_text, predicted_colors, color_scores


def _query_pills(
    *,
    shape: str | None,
    colors: list[str],
    imprint: str | None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    from smart_drug_assistant.db.connection import get_db_connection

    where = []
    params: list[Any] = []

    if shape:
        where.append("UPPER(shape) = UPPER(%s)")
        params.append(shape)

    # Filter by colors. If multiple colors are predicted, treat them as alternatives (OR),
    # otherwise we would incorrectly require the DB color string to contain all colors.
    color_terms = [c for c in colors if c]
    if color_terms:
        color_clauses = []
        for c in color_terms:
            color_clauses.append("color ILIKE %s")
            params.append(f"%{c}%")
        where.append("(" + " OR ".join(color_clauses) + ")")

    # Fuzzy imprint via regex (in addition to Python ranking)
    pred_tokens = _tokenize_predicted_imprint(imprint or "")
    regex_parts = [_confusable_regex_token(t) for t in pred_tokens if _confusable_regex_token(t)]
    if regex_parts:
        where.append("imprint ~* %s")
        params.append("(" + "|".join(regex_parts) + ")")

    sql = "SELECT image_path, image_filename, pill_name, shape, color, imprint FROM pills"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " LIMIT %s"
    params.append(limit)

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        finally:
            try:
                cur.close()
            except Exception:
                pass
        results = []
        for r in rows:
            results.append(
                {
                    "image_path": r[0],
                    "image_filename": r[1],
                    "pill_name": r[2],
                    "shape": r[3],
                    "color": r[4],
                    "imprint": r[5],
                }
            )
        return results
    finally:
        conn.close()


def _get_pills_count() -> int:
    from smart_drug_assistant.db.connection import get_db_connection

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM pills")
            return int(cur.fetchone()[0])
        finally:
            try:
                cur.close()
            except Exception:
                pass
    finally:
        conn.close()


def _rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    predicted_imprint: str,
    predicted_shape: str,
    chosen_colors: list[str],
) -> list[dict[str, Any]]:
    pred_tokens = _tokenize_predicted_imprint(predicted_imprint)
    for c in candidates:
        tokens = _tokenize_db_imprint(c.get("imprint", ""))
        best = 0.0
        best_token = ""
        best_pred_token = ""
        # Score by best predicted-token vs db-token similarity.
        for pt in (pred_tokens or [""]):
            for t in tokens:
                s = _similarity(pt, t)
                if s > best:
                    best = s
                    best_token = t
                    best_pred_token = pt
        c["match"] = {
            "imprint_similarity": round(best, 3),
            "best_predicted_token": best_pred_token,
            "best_imprint_token": best_token,
            "shape_match": bool(predicted_shape) and str(c.get("shape", "")).upper() == str(predicted_shape).upper(),
            "color_match_count": sum(1 for col in chosen_colors if col and col.lower() in str(c.get("color", "")).lower()),
        }
    return sorted(
        candidates,
        key=lambda x: (
            x.get("match", {}).get("imprint_similarity", 0.0),
            x.get("match", {}).get("color_match_count", 0),
            1 if x.get("match", {}).get("shape_match", False) else 0,
        ),
        reverse=True,
    )

# --- ENDPOINT ---
@router.post("/pill_identify/")
async def pill_identify(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # YOLO Shape Prediction
    results = shape_model(image)
    probs = results[0].probs
    top_idx = int(probs.top1)
    shape_pred = results[0].names[top_idx]
    shape_conf = float(probs.top1conf.item().__round__(2))

    # Multitask Pill Prediction
    imprint_pred, color_pred, color_scores = predict_multitask(image)

    chosen_colors = _choose_color_filters(color_scores)

    matches: list[dict[str, Any]] = []
    db_error: str | None = None
    pills_count: int | None = None
    try:
        pills_count = _get_pills_count()

        has_imprint = bool(_tokenize_predicted_imprint(imprint_pred))

        # 1) strict: if we have an imprint, do NOT filter by color first.
        # Color prediction is noisier than imprint OCR, and premature color filtering
        # can remove the correct pill from the candidate set.
        if has_imprint:
            matches = _query_pills(shape=shape_pred, colors=[], imprint=imprint_pred, limit=120)
        else:
            matches = _query_pills(shape=shape_pred, colors=chosen_colors, imprint=None, limit=120)

        # 2) add color filter if needed
        if not matches:
            matches = _query_pills(shape=shape_pred, colors=chosen_colors, imprint=imprint_pred if has_imprint else None, limit=120)

        # 3) relax: drop imprint
        if not matches:
            matches = _query_pills(shape=shape_pred, colors=chosen_colors, imprint=None, limit=120)

        # 4) relax: drop colors
        if not matches:
            matches = _query_pills(shape=shape_pred, colors=[], imprint=None, limit=120)

        # 5) relax: drop shape (useful when YOLO is uncertain or predicts a shape not present in DB)
        if not matches and shape_conf < 0.30:
            matches = _query_pills(shape=None, colors=chosen_colors, imprint=None, limit=120)
        if not matches:
            matches = _query_pills(shape=None, colors=[], imprint=None, limit=120)

        matches = _rank_candidates(
            matches,
            predicted_imprint=imprint_pred,
            predicted_shape=shape_pred,
            chosen_colors=chosen_colors,
        )[:10]
    except Exception as e:
        db_error = str(e)

    return JSONResponse(
        {
            "shape": {"class": shape_pred, "confidence": shape_conf},
            "imprint": imprint_pred,
            "colors": list(color_pred),
            "color_scores": {k: round(v, 4) for k, v in sorted(color_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]},
            "db_query": {
                "chosen_colors": chosen_colors,
                "pills_count": pills_count,
                "error": db_error,
            },
            "matches": matches,
        }
    )
