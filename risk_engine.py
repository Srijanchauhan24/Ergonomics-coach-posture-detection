import time

bad_duration = 0
last_alert = 0

MODERATE = 120
HIGH = 300
COOLDOWN = 20

def get_risk(posture, frame_time):
    global bad_duration, last_alert

    now = time.time()

    if posture != "Good":
        bad_duration += frame_time
    else:
        bad_duration = max(0, bad_duration - frame_time * 0.7)

    if bad_duration < MODERATE:
        risk = "LOW"
    elif bad_duration < HIGH:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    alert = None

    if risk == "MODERATE" and now - last_alert > COOLDOWN:
        alert = "Warning! Fix posture"
        last_alert = now

    if risk == "HIGH" and now - last_alert > COOLDOWN:
        alert = "High Risk! Take a break"
        last_alert = now

    return risk, bad_duration, alert