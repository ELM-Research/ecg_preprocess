import xml.etree.ElementTree as ET
import numpy as np
from base64 import b64decode

from configs.constants import PTB_ORDER

def to_text(el: ET.Element | None) -> str | None:
    return el.text.strip() if el is not None and el.text else None

def to_int(s: str | None) -> int | None:
    try:
        return int(s) if s is not None else None
    except ValueError:
        return None

def parse_sf(root: ET.Element):
    sb = int(root.findtext('.//Waveform/SampleBase'))
    se = int(root.findtext('.//Waveform/SampleExponent'))
    return int(sb * (2 ** se))

def parse_report(root: ET.Element) -> tuple[str | None, list[str]]:
    def _smart_append(acc: list[str], txt: str) -> None:
        if not acc:
            acc.append(txt)
            return
        prev = acc[-1]
        if txt and txt[0] in ",;:.%)":
            acc[-1] = prev + txt
        elif prev.endswith(("-", "/")):
            acc[-1] = prev + txt
        else:
            acc[-1] = prev + " " + txt
    diag = root.find(".//Diagnosis")
    out: list[str] = []
    line_buf: list[str] = []
    for node in diag.findall("DiagnosisStatement"):
        txt = to_text(node.find("StmtText"))
        if txt:
            _smart_append(line_buf, txt)
        flag = (node.findtext("StmtFlag") or "").strip().upper()
        if flag == "ENDSLINE" and line_buf:
            out.append(line_buf[-1].strip())
            line_buf = []

    if line_buf:
        out.append(line_buf[-1].strip())
    diagnosis = "; ".join(out)
    return diagnosis


def global_interval_checker(xml_path: str, max_abs: int = 10000) -> bool:
    root = ET.parse(xml_path).getroot()
    meas = root.find('./RestingECGMeasurements')
    P_on = to_int(to_text(meas.find("POnset")))
    P_off = to_int(to_text(meas.find("POffset")))
    Q_on = to_int(to_text(meas.find("QOnset")))
    Q_off = to_int(to_text(meas.find("QOffset")))
    T_off = to_int(to_text(meas.find("TOffset")))
    pairs = [
        ("P",   P_on, P_off),
        ("QRS", Q_on, Q_off),
        ("T",   Q_off, T_off),
    ]
    for _, a, b in pairs:
        if a is None or b is None:
            continue
        if abs(a) > max_abs or abs(b) > max_abs:
            return True
        if a >= b:
            return True
        if (b - a) > max_abs:
            return True
    return False

def parse_ecg(root):
    leads = {}
    for ld in root.findall(".//LeadData"):
        lead_id = ld.find("LeadID").text
        data = b64decode(ld.find("WaveFormData").text)
        leads[lead_id] = np.frombuffer(data, dtype="<i2")
    leads["III"] = leads["II"] - leads["I"]
    leads["aVR"] = -0.5 * (leads["I"] + leads["II"])
    leads["aVL"] = 0.5 * (leads["I"] - leads["III"])
    leads["aVF"] = 0.5 * (leads["II"] + leads["III"])
    ecg = np.array([leads[ld] for ld in PTB_ORDER])
    return np.transpose(ecg, (1, 0))