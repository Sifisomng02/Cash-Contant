# =============================================================================
# CashContant — Central Configuration
# All file paths, constants, and app settings live here.
# =============================================================================

import os

# --- Base Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# --- Data File Paths ---
WORLD_BANK_FILE         = os.path.join(DATA_DIR, "API_FB.ATM.TOTL.P5_DS2_en_csv_v2_6160.csv")
ATM_FILE                = os.path.join(DATA_DIR, "atm_withdrawals.csv")
SMARTCASH_FILE          = os.path.join(DATA_DIR, "smartcash_southafrica_dataset.csv")
EXCEL_FILE              = os.path.join(DATA_DIR, "money-and-banking.xlsx")
BRANCH_LOCATIONS_FILE   = os.path.join(DATA_DIR, "branch_locations.csv")
CASH_CIRCULATION_FILE   = os.path.join(DATA_DIR, "cash_in_circulation.csv")

# --- Image Paths ---
LOGO_FILE = os.path.join(IMAGES_DIR, "logo.png")

# --- App Settings ---
APP_TITLE       = "CashContant"
APP_ICON        = "💡"
APP_AUTHOR      = "Sifiso Mnguni"
APP_VERSION     = "2.0.0"

# --- Forecast Settings ---
FORECAST_DEFAULT_DAYS   = 90
FORECAST_MIN_DAYS       = 30
FORECAST_MAX_DAYS       = 365

# --- Security Thresholds ---
RAPID_WITHDRAWAL_SECONDS    = 60    # Withdrawals < 60s apart are suspicious
IMPOSSIBLE_TRAVEL_SECONDS   = 600   # Same card, different ATM < 10min apart
AFTER_HOURS_START           = 0     # Midnight
AFTER_HOURS_END             = 4     # 4am
HIGH_VALUE_PERCENTILE       = 0.995 # Top 0.5% = high-value outlier

# --- SARB Key Stats (from 2023 Payments Report) ---
SARB_STATS = {
    "cash_users_pct": 82,
    "small_txn_cash_pct": 58,
    "digital_fee_barrier_pct": 34,
    "atm_withdrawals_2022_trillion": 1.2,
}
