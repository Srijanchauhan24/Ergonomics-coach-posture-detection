import time

bad_posture_duration = 0
risk_level = "LOW"
last_alert_time = 0

FATIGUE_MODERATE = 120
FATIGUE_HIGH = 300
COOLDOWN = 20

def risk_engine(current_posture, frame_time):
    global bad_posture_duration, risk_level, last_alert_time

    now = time.time()

    if current_posture != "Good":
        bad_posture_duration += frame_time
    else:
        bad_posture_duration = max(0, bad_posture_duration - frame_time * 0.7)

    if bad_posture_duration < FATIGUE_MODERATE:
        risk_level = "LOW"
    elif bad_posture_duration < FATIGUE_HIGH:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    alert = None

    if risk_level == "MODERATE" and now - last_alert_time > COOLDOWN:
        alert = "Warning! Fix your posture."
        last_alert_time = now

    if risk_level == "HIGH" and now - last_alert_time > COOLDOWN:
        alert = "High Risk! Take a break!"
        last_alert_time = now

    return risk_level, bad_posture_duration, alert