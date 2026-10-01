"""Static definitions shared by the AURA simulator, gateway and edge nodes.

AURA = Adaptive Uncertainty-Regulated Ambient intelligence.
"""
import numpy as np

STEP_S = 30                      # sensing period of every edge node (seconds)
STEPS_PER_DAY = 24 * 3600 // STEP_S

ACTIVITIES = ["SLEEP", "HYGIENE", "COOK", "EAT", "RELAX", "WORK", "AWAY", "FALL"]
K = len(ACTIVITIES)
SLEEP, HYGIENE, COOK, EAT, RELAX, WORK, AWAY, FALL = range(K)

SENSORS = ["pir_living", "pir_kitchen", "pir_bedroom", "pir_bath", "power_kw",
           "co2_ppm", "sound_db", "water_lpm", "light_lux", "wear_accel_g",
           "bed_load_kg"]
S = len(SENSORS)

# Activity-conditioned steady-state targets (rows: activity, cols: sensor).
TARGETS = np.array([
    #  pirL pirK pirB pirBa  kW    CO2   dB   water lux  accel  bed
    [0.0, 0.0, 0.3, 0.0, 0.08, 900, 28, 0.0, 2, 0.02, 70],   # SLEEP
    [0.2, 0.1, 0.3, 4.0, 0.30, 650, 50, 4.0, 300, 0.25, 0],  # HYGIENE
    [0.5, 4.0, 0.0, 0.0, 2.00, 800, 55, 1.5, 400, 0.35, 0],  # COOK
    [1.5, 1.0, 0.0, 0.0, 0.20, 750, 45, 0.1, 300, 0.15, 0],  # EAT
    [2.0, 0.1, 0.0, 0.0, 0.35, 700, 55, 0.0, 150, 0.05, 0],  # RELAX
    [0.6, 0.1, 2.0, 0.0, 0.25, 750, 38, 0.0, 450, 0.10, 0],  # WORK
    [0.0, 0.0, 0.0, 0.0, 0.06, 450, 25, 0.0, 20, 0.30, 0],   # AWAY
    [0.1, 0.0, 0.0, 0.1, 0.15, 600, 33, 0.0, 200, 0.01, 0],  # FALL
], dtype=float)

# First-order lag coefficient (fraction of the gap closed per step).
LAG = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.03, 0.5, 0.6, 0.6, 0.5, 0.7])
# Measurement-noise standard deviation (absolute units).
NOISE = np.array([0.9, 0.9, 0.9, 0.9, 0.25, 12.0, 7.0, 0.9, 80.0, 0.09, 4.0])
# Slow, activity-independent drift (AR(1)) std that the twin predictor can track.
DRIFT = np.array([0.3, 0.3, 0.3, 0.3, 0.1, 40.0, 4.0, 0.3, 60.0, 0.04, 3.0])
# Lower physical bounds.
FLOOR = np.zeros(S)

FALL_SPIKE_G = 3.0               # wearable impact at fall onset
BENIGN_SPIKE_P = 0.002           # prob./step of a benign impact (sitting down hard, dropped device)
BENIGN_SPIKE_G = 2.2

# Sensors whose information is safety-critical (used by the AURA safety floor).
SAFETY_CLASS = FALL

# Radio / energy model (IEEE 802.15.4-class node at 0 dBm, 3 V CR2032 coin cell).
E_TX_MJ = 0.90           # one uplink packet incl. wake-up, CSMA/CA and ACK window
E_RX_BEACON_MJ = 0.45    # receiving one downlink attention beacon
E_SENSE_MJ = 0.020       # sampling + twin-predictor update per step
P_SLEEP_UW = 6.0         # deep-sleep floor (2 uA @ 3 V)
BATTERY_J = 0.225 * 3.0 * 3600 * 0.8   # usable energy of a CR2032 (80 % derating)
