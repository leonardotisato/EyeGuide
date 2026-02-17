import numpy as np

# Definisci gli intervalli di confidenza per "student" e "distilled student"
student_confidence_intervals = {
    "accuracy": (0.7669, 0.8447),
    "precision": (0.7808, 0.9270),
    "recall": (0.6778, 0.8412),
    "f1_score": (0.6986, 0.8625)
}

distilled_student_confidence_intervals = {
    "accuracy": (0.7895, 0.9098),
    "precision": (0.7895, 0.9175),
    "recall": (0.7490, 0.8932),
    "f1_score": (0.7637, 0.8991)
}

# Funzione per verificare se c'è sovrapposizione tra due intervalli
def has_overlap(interval1, interval2):
    return not (interval1[1] < interval2[0] or interval2[1] < interval1[0])

# Confronta gli intervalli di confidenza
for metric in student_confidence_intervals:
    student_interval = student_confidence_intervals[metric]
    distilled_student_interval = distilled_student_confidence_intervals[metric]
    
    if has_overlap(student_interval, distilled_student_interval):
        print(f"Per la metrica '{metric}': C'è sovrapposizione tra gli intervalli di confidenza.")
    else:
        print(f"Per la metrica '{metric}': Non c'è sovrapposizione tra gli intervalli di confidenza. Differenza significativa.")
