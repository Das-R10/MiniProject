# check_accuracy.py
# Run: python check_accuracy.py results.json
#
# Ground truth mapped to the ACTUAL sample_contract.txt structure.

import json, sys

# ─────────────────────────────────────────────────────────────────
# GROUND TRUTH — actual contract clause IDs and expected labels
# ─────────────────────────────────────────────────────────────────
GROUND_TRUTH = {
    # Section 1 — Definitions (hard-neutral)
    "1"   : "Neutral",

    # Section 2 — Commencement and Probation
    "2.1" : "Neutral",
    "2.2" : "Con",      # "without notice and without any compensation whatsoever"
    "2.3" : "Con",      # "sole discretion of the Management"

    # Section 3 — Compensation and Benefits
    "3.1" : "Pro",      # salary entitlement
    "3.2" : "Pro",      # Provident Fund / EPF Act
    "3.3" : "Pro",      # ESIC coverage
    "3.4" : "Pro",      # Gratuity Act
    "3.5" : "Con",      # "sole discretion at any time without prior notice"
    "3.6" : "Con",      # "absolute discretion" + "no claim shall lie"

    # Section 4 — Working Hours and Leave
    "4.1" : "Pro",      # earned leave entitlement
    "4.2" : "Pro",      # sick/casual leave
    "4.3" : "Pro",      # maternity benefit act
    "4.4" : "Con",      # "sole discretion" overtime without pay
    "4.5" : "Con",      # "subject to management approval at its discretion"

    # Section 5 — Termination
    "5.1" : "Pro",      # 30 days prior written notice by either party
    "5.2" : "Con",      # "immediately and without notice" + "without compensation"
    "5.3" : "Con",      # "at any time, for any reason or no reason, at sole discretion"
    "5.4" : "Pro",      # retrenchment compensation / IDA reference
    "5.5" : "Con",      # "forfeit any pending dues without further notice"

    # Section 6 — Confidentiality
    "6.1" : "Neutral",
    "6.2" : "Neutral",
    "6.3" : "Neutral",
    "6.4" : "Con",      # injunctive relief "without any limitation"

    # Section 7 — Intellectual Property
    "7.1" : "Con",      # IP grab including "outside working hours"
    "7.2" : "Con",      # irrevocable assignment "without additional compensation"
    "7.3" : "Con",      # waives all moral rights

    # Section 8 — Non-Compete and Non-Solicitation
    "8.1" : "Con",      # 12-month non-compete
    "8.2" : "Con",      # 24-month non-solicitation
    "8.3" : "Neutral",  # acknowledgement clause — no strong signal

    # Section 9 — Disciplinary and Grievance
    "9.1" : "Con",      # disciplinary policy at "sole discretion"
    "9.2" : "Con",      # suspend "with or without pay" at discretion
    "9.3" : "Pro",      # employee has right to raise grievance
    "9.4" : "Pro",      # domestic inquiry required before disciplinary action

    # Section 10 — Governing Law (hard-neutral)
    "10"  : "Neutral",

    # Section 11 — Entire Agreement (hard-neutral)
    "11"  : "Neutral",

    # Section 12 — Amendments
    "12.1": "Con",      # "unilaterally amend...without the consent of the Employee"
    "12.2": "Con",      # continued employment = deemed acceptance
}

# ── Priority groups ────────────────────────────────────────────────
CRITICAL_CON = {"5.2", "5.3", "12.1"}          # must be flagged Con
HIGH_CON     = {"2.2", "3.5", "4.4", "7.1", "7.2", "9.2"}
HARD_NEUTRAL = {"1", "10", "11"}

# ─────────────────────────────────────────────────────────────────
# Load results.json
# ─────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python check_accuracy.py results.json")
    sys.exit(1)

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

results  = data.get("results", data)
predicted = {str(r["clause_id"]).strip(): r["label"].strip().capitalize() for r in results}

# ─────────────────────────────────────────────────────────────────
# Score
# ─────────────────────────────────────────────────────────────────
correct = wrong = not_found = 0
details = []

for cid, expected in GROUND_TRUTH.items():
    pred = predicted.get(cid)
    if pred is None:
        not_found += 1
        details.append((cid, expected, "NOT FOUND", "❓"))
    elif pred == expected:
        correct += 1
        details.append((cid, expected, pred, "✅"))
    else:
        wrong += 1
        details.append((cid, expected, pred, "❌"))

scored      = correct + wrong
overall_pct = round(correct / scored * 100, 1) if scored else 0

con_ids  = [c for c, e in GROUND_TRUTH.items() if e == "Con"]
pro_ids  = [c for c, e in GROUND_TRUTH.items() if e == "Pro"]
neu_ids  = [c for c, e in GROUND_TRUTH.items() if e == "Neutral"]

con_correct = sum(1 for c in con_ids  if predicted.get(c) == "Con")
pro_correct = sum(1 for c in pro_ids  if predicted.get(c) == "Pro")
neu_correct = sum(1 for c in neu_ids  if predicted.get(c) == "Neutral")

crit_correct        = sum(1 for c in CRITICAL_CON if predicted.get(c) == "Con")
high_correct        = sum(1 for c in HIGH_CON     if predicted.get(c) == "Con")
hard_neutral_correct= sum(1 for c in HARD_NEUTRAL if predicted.get(c) == "Neutral")

# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────
S  = "─" * 65
S2 = "═" * 65

print(f"\n{S2}")
print(f"  LEXANALYZE ACCURACY REPORT")
print(f"{S2}\n")
print(f"  Overall accuracy   : {correct}/{scored} = {overall_pct}%")
print(f"  Clauses not found  : {not_found}\n")
print(f"  ── By label ──────────────────────────────────────────")
print(f"  Con     : {con_correct}/{len(con_ids)}  ({round(con_correct/len(con_ids)*100,1)}%)")
print(f"  Pro     : {pro_correct}/{len(pro_ids)}  ({round(pro_correct/len(pro_ids)*100,1)}%)")
print(f"  Neutral : {neu_correct}/{len(neu_ids)}  ({round(neu_correct/len(neu_ids)*100,1)}%)\n")
print(f"  ── Priority checks ───────────────────────────────────")
print(f"  Critical CON (5.2, 5.3, 12.1)    : {crit_correct}/3  {'✅ PASS' if crit_correct==3 else '❌ FAIL'}")
print(f"  High CON     (2.2,3.5,4.4,7.1..) : {high_correct}/6  {'✅ PASS' if high_correct>=5 else '⚠ PARTIAL' if high_correct>=3 else '❌ FAIL'}")
print(f"  Hard-neutral (Sec 1, 10, 11)      : {hard_neutral_correct}/3  {'✅ PASS' if hard_neutral_correct==3 else '❌ FAIL'}\n")

if overall_pct >= 80 and crit_correct == 3:
    grade = "A  — Excellent"
elif overall_pct >= 65 and crit_correct >= 2:
    grade = "B  — Good"
elif overall_pct >= 50 and crit_correct >= 1:
    grade = "C  — Acceptable"
else:
    grade = "D  — Needs tuning"
print(f"  Grade : {grade}\n")

print(f"{S}")
print(f"  {'ID':<8} {'Expected':<10} {'Predicted':<12} {'Match'}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*5}")
for cid, expected, pred, icon in details:
    print(f"  {cid:<8} {expected:<10} {pred:<12} {icon}")

print(f"\n{S}")
print(f"  MISMATCHES")
print(f"{S}")
bad = [(c,e,p) for c,e,p,i in details if i in ("❌","❓")]
if bad:
    for cid, exp, pred in bad:
        tag = "⚠ CRITICAL" if cid in CRITICAL_CON else ("⚠ HIGH" if cid in HIGH_CON else "")
        print(f"  {cid:<8} expected={exp:<9} got={pred:<12} {tag}")
else:
    print("  Perfect score!")
print(f"\n{S2}\n")