# app/ocr.py
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageOps
import pytesseract

from .models import ContractExtraction

# ---------------- Tesseract path ----------------
pytesseract.pytesseract.tesseract_cmd = (
    os.getenv("TESSERACT_CMD") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------------- PDF -> Images ----------------
def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    images: List[Image.Image] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


# ---------------- OCR ----------------
def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Light preprocessing to reduce OCR merge errors:
    - grayscale
    - autocontrast
    """
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    return gray


def ocr_image(img: Image.Image) -> str:
    # psm 6 usually works well for these "semi-structured" PDFs
    img = _preprocess_for_ocr(img)
    return pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 6")


# ---------------- Helpers ----------------
DATE_PATTERNS = [
    "%B %d, %Y",  # November 19, 2025
    "%b %d, %Y",  # Nov 19, 2025
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
]

def _norm_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def _collapse_spaces(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return re.sub(r"\s+", " ", s).strip()

def _digits_only(s: Optional[str]) -> str:
    return re.sub(r"\D", "", s or "")

def _first_group(pattern: str, text: str, flags=0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

def _money_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", "")
    m = re.search(r"(\d+(?:\.\d{1,2})?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def _parse_date_str(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = _collapse_spaces(s)
    if not s:
        return None

    # strip trailing punctuation / label fragments
    s = re.sub(r"[;|]+$", "", s).strip()

    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # fallback: try ISO-ish
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _extract_section(text: str, start_marker: str, end_markers: List[str]) -> str:
    """
    Best-effort section extractor: returns text between start_marker and the earliest end_marker.
    """
    start = re.search(re.escape(start_marker), text, re.IGNORECASE)
    if not start:
        return ""

    start_pos = start.end()
    end_pos = len(text)

    for em in end_markers:
        m = re.search(re.escape(em), text[start_pos:], re.IGNORECASE)
        if m:
            end_pos = min(end_pos, start_pos + m.start())

    return text[start_pos:end_pos].strip()


def _extract_value_by_label(text: str, label: str, *, max_len: int = 200) -> Optional[str]:
    """
    Extracts "Label: value" where value can be on the same line.
    Stops at newline.
    """
    # Example: "Order Number: 151687471"
    pat = rf"{re.escape(label)}\s*:\s*([^\n]+)"
    v = _first_group(pat, text, flags=re.IGNORECASE)
    if not v:
        return None
    v = v.strip()
    if len(v) > max_len:
        v = v[:max_len].strip()
    return v


def _extract_block_after_label(text: str, label: str, stop_labels: List[str]) -> Optional[str]:
    """
    Extract a block after "label:" and continue across lines until a stop label is encountered.
    Used for multi-line Address extraction.
    """
    stop_union = "|".join(re.escape(x) for x in stop_labels)
    # capture everything after label: until a stop label (as a field) or end
    pat = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n(?:{stop_union})\s*:|\n[A-Z][A-Z ]{{3,}}:|\Z)"
    v = _first_group(pat, text, flags=re.IGNORECASE | re.DOTALL)
    return _collapse_spaces(v) if v else None


def _extract_phone_from_chunk(chunk: str) -> Optional[str]:
    """
    Return a 10-digit phone number from chunk; prioritize first match.
    """
    if not chunk:
        return None

    # Common: (780) 617-4431 OR 780-617-4431 OR 780 617 4431
    m = re.search(r"(\(?\d{3}\)?\s*[- ]?\s*\d{3}\s*[- ]?\s*\d{4})", chunk)
    if not m:
        return None

    digits = _digits_only(m.group(1))
    if len(digits) >= 10:
        return digits[-10:]  # safe in case OCR adds country code
    return None


def _extract_long_digit_run(chunk: str, min_len: int, max_len: int) -> Optional[str]:
    """
    Find the longest digit-run within bounds.
    """
    if not chunk:
        return None
    digits = _digits_only(chunk)
    runs = re.findall(rf"\d{{{min_len},{max_len}}}", digits)
    if not runs:
        return None
    return max(runs, key=len)


def _extract_labeled_date(text: str, label: str) -> Optional[datetime]:
    """
    More robust than "between labels":
    Finds label: <date> even if text continues after the date on same line.
    """
    # Example: "End Date: November 18, 2027 Early Cancellation Fee(s): ..."
    pat = rf"{re.escape(label)}\s*:\s*([A-Za-z]{{3,9}}\s+\d{{1,2}},\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}})"
    raw = _first_group(pat, text, flags=re.IGNORECASE)
    return _parse_date_str(raw)


def _pick_customer_name(info_section: str) -> Optional[str]:
    """
    Dynamic rules for customer identifier:
    Priority:
    1) Customer Name
    2) Company Name
    3) Customer ID  (your sp_contract_print_out expects an ID-style value)
    4) Account Number
    """
    for label in ["Customer Name", "Company Name", "Customer ID", "Account Number"]:
        v = _extract_value_by_label(info_section, label)
        v = _collapse_spaces(v)
        if v:
            # strip trailing "First Bill Date..." / "Monthly Payment Method..." if OCR merged
            v = re.split(
                r"(Monthly Payment Method|First Bill Date|User Name|Account Number|Phone Number|Default Voicemail Password|Address)\s*:",
                v,
                flags=re.IGNORECASE
            )[0].strip()
            if v:
                return v
    return None


def _pick_plan_name_and_charge(rate_section: str, full_text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Supports:
    - Plan: SmartPay Tab 20GB Monthly Rate Plan Charge: $20.00
    - Plan: EPP BYOD 60GB Lite  Minimum Monthly Charge: $85.00
    """
    # First: get the Plan line (most reliable for name)
    plan_line = _extract_value_by_label(rate_section, "Plan", max_len=250) or _extract_value_by_label(full_text, "Plan", max_len=250)

    plan_name: Optional[str] = None
    plan_charge: Optional[float] = None

    if plan_line:
        # Plan charge: try Monthly Rate Plan Charge
        charge_raw = _first_group(r"Monthly Rate Plan Charge\s*:\s*\$?\s*([0-9.,]+)", plan_line, flags=re.IGNORECASE)
        if charge_raw:
            plan_charge = _money_to_float(charge_raw)

        # If not found, try Minimum Monthly Charge (BYOD header style)
        if plan_charge is None:
            charge_raw = _first_group(r"Minimum Monthly Charge\s*:\s*\$?\s*([0-9.,]+)", plan_line, flags=re.IGNORECASE)
            if charge_raw:
                plan_charge = _money_to_float(charge_raw)

        # Plan name: everything before "Monthly Rate Plan Charge" or "Minimum Monthly Charge"
        name_part = re.split(r"Monthly Rate Plan Charge\s*:|Minimum Monthly Charge\s*:", plan_line, flags=re.IGNORECASE)[0].strip()
        plan_name = _collapse_spaces(name_part)

    # Fallback: if plan name is still missing or clearly not a name,
    # use the first descriptive bullet/line after Plan in the rate section.
    if not plan_name:
        # Pick first non-empty line in rate section that is not a header and not a label
        lines = [ln.strip("•* \t") for ln in rate_section.split("\n") if ln.strip()]
        for ln in lines:
            if re.search(r"^(YOUR RATE PLAN DETAILS|YOUR RATE PLAN ADD-ONS|MINIMUM MONTHLY CHARGE|TOTAL MONTHLY CHARGE)", ln, re.IGNORECASE):
                continue
            if re.search(r"^(Plan\s*:)", ln, re.IGNORECASE):
                continue
            # likely description line
            plan_name = _collapse_spaces(ln)
            break

    return plan_name, plan_charge


# ---------------- Parsing ----------------
def parse_contract_text(full_text: str) -> ContractExtraction:
    text = _norm_text(full_text)

    # Targeted sections (prevents Store Phone Number bleeding into customer phone)
    info_section = _extract_section(
        text,
        "YOUR INFORMATION:",
        ["YOUR DEVICE DETAILS:", "YOUR DEVICE DETAILS", "YOUR RATE PLAN DETAILS:", "CRITICAL INFORMATION SUMMARY"]
    )

    device_section = _extract_section(
        text,
        "YOUR DEVICE DETAILS:",
        ["YOUR RATE PLAN DETAILS:", "YOUR RATE PLAN DETAILS", "MINIMUM MONTHLY CHARGE", "TOTAL MONTHLY CHARGE"]
    )

    rate_section = _extract_section(
        text,
        "YOUR RATE PLAN DETAILS:",
        ["YOUR RATE PLAN ADD-ONS:", "YOUR PROMOTIONS:", "TOTAL MONTHLY CHARGE:", "ONE-TIME CHARGES:"]
    )

    # --- Customer Name (Customer Name OR Company Name OR Customer ID) ---
    customer_name = _pick_customer_name(info_section or text)

    # --- Customer Phone (from YOUR INFORMATION only) ---
    raw_phone_line = _extract_value_by_label(info_section or text, "Phone Number") or _extract_value_by_label(info_section or text, "Phone")
    customer_phone = _extract_phone_from_chunk(raw_phone_line or "") if raw_phone_line else None

    # --- Customer Address (multi-line) ---
    # stop labels are the next fields typically following Address in YOUR INFORMATION
    customer_address = _extract_block_after_label(
        info_section or text,
        "Address",
        ["Monthly Payment Method", "First Bill Date", "User Name", "Account Number", "Phone Number", "Default Voicemail Password", "Customer ID"]
    )

    # --- Order Number / Activity (from Critical Info area; safe on whole text) ---
    order_number = _extract_value_by_label(text, "Order Number")
    if order_number:
        # remove OCR merge: "Order Number: 152... Store: ..."
        order_number = re.split(r"\b(Store|Date|Activity)\s*:", order_number, flags=re.IGNORECASE)[0].strip()
    order_number = _collapse_spaces(order_number)

    activity = _extract_value_by_label(text, "Activity")
    if activity:
        activity = re.split(r"\b(Store Phone Number|Store|Date)\s*:", activity, flags=re.IGNORECASE)[0].strip()
    activity = _collapse_spaces(activity)

    # --- Dates (robust label->date extractor) ---
    # Prefer within device section first (often has Commitment Period block),
    # fallback to whole text.
    contract_start = _extract_labeled_date(device_section or text, "Start Date") or _extract_labeled_date(text, "Start Date")
    contract_end = _extract_labeled_date(device_section or text, "End Date") or _extract_labeled_date(text, "End Date")

    # --- Plan name & plan charge ---
    plan_name, plan_charge = _pick_plan_name_and_charge(rate_section or text, text)

    # --- Minimum Monthly Plan (SmartPay docs) ---
    minimum_monthly_plan = _money_to_float(
        _first_group(
            r"MINIMUM MONTHLY CHARGE\s*\(FOR DEVICE AND RATE PLAN\)\s*:\s*\$?\s*([0-9.,]+)",
            text,
            flags=re.IGNORECASE,
        )
    )

    # If plan_charge still missing:
    # BYOD commonly has "Minimum Monthly Charge: $85.00" elsewhere
    if plan_charge is None:
        byod_min = _money_to_float(_first_group(r"Minimum Monthly Charge\s*:\s*\$?\s*([0-9.,]+)", text, flags=re.IGNORECASE))
        if byod_min is not None:
            plan_charge = byod_min
        else:
            # SmartPay: "Monthly Rate Plan Charge: $20.00" may appear elsewhere
            mrc = _money_to_float(_first_group(r"Monthly Rate Plan Charge\s*:\s*\$?\s*([0-9.,]+)", text, flags=re.IGNORECASE))
            if mrc is not None:
                plan_charge = mrc

    # --- Device model / IMEI / SIM ---
    model_raw = _extract_value_by_label(device_section or text, "Model", max_len=200)
    if model_raw:
        # cut if OCR merged "Early Cancellation Fee(s)" etc
        model_raw = re.split(r"\b(Early Cancellation Fee|IMEI/ESN/MEID|SIM Number|Commitment Period|Start Date|End Date)\b", model_raw, flags=re.IGNORECASE)[0].strip()
    device_model = _collapse_spaces(model_raw)

    imei_chunk = _extract_value_by_label(device_section or text, "IMEI/ESN/MEID", max_len=200) or _extract_value_by_label(device_section or text, "IMEI", max_len=200)
    device_imei = _extract_long_digit_run(imei_chunk, min_len=10, max_len=20)

    sim_chunk = _extract_value_by_label(device_section or text, "SIM Number", max_len=250) or _extract_value_by_label(device_section or text, "SIM", max_len=250)
    sim_number = _extract_long_digit_run(sim_chunk, min_len=18, max_len=22)

    return ContractExtraction(
        customer_name=customer_name,
        customer_phone=customer_phone,
        customer_address=customer_address,
        plan_name=plan_name,
        plan_charge=plan_charge,
        minimum_monthly_plan=minimum_monthly_plan,
        contract_start_date=contract_start.date() if contract_start else None,
        contract_end_date=contract_end.date() if contract_end else None,
        order_number=order_number,
        activity=activity,
        down_payment=None,
        device_model=device_model,
        device_imei=device_imei,
        sim_number=sim_number,
        raw_text=full_text,
    )


# ---------------- Pipeline ----------------
def extract_from_pdf(pdf_bytes: bytes) -> ContractExtraction:
    images = pdf_to_images(pdf_bytes)
    full_text = "\n".join(ocr_image(img) for img in images)
    return parse_contract_text(full_text)
