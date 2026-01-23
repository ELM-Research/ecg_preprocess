# DATA_DIR = "../data"
DATA_DIR = "../ECG-Bench/ecg_bench/data"

LOG_DIR = "./.logs"

PTB_ORDER = ["I", "II", "III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

V_PACING_PATTERNS = ("v paced", "v-paced", "ventricular paced", 
                     "ventricularly paced", "ventricular paced",
                     "ventricular pacing", "ventricularly pacing",
                     "v-pacing", "v pacing", "v pace", "v-pace",
                     "ventricular pace", "ventricularly pace", )

LBBB_PATTERNS = ("left bundle branch block", "lbbb",)

QRS_DUR_WIDE_PATTERNS = ("wide",)

### Name to Pattern Matching
BATCH_LABEL_DICT = {"v_pacing": V_PACING_PATTERNS, 
                   "lbbb": LBBB_PATTERNS, 
                   "qrs_dur_wide": QRS_DUR_WIDE_PATTERNS}