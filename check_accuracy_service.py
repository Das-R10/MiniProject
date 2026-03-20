# check_accuracy_service.py
# Run: python check_accuracy_service.py results.json
#
# Ground truth for service_agreement.txt
# Upload service_agreement.txt to LexAnalyze, download JSON, then run this.

import json, sys

GROUND_TRUTH = {
    # Section 1 — Definitions (hard-neutral)
    "1"    : "Neutral",
    "1.1"  : "Neutral",
    "1.2"  : "Neutral",
    "1.3"  : "Neutral",
    "1.4"  : "Neutral",
    "1.5"  : "Neutral",

    # Section 2 — Scope of Services
    "2.1"  : "Pro",      # professional standard commitment
    "2.2"  : "Pro",      # advance notice of personnel changes
    "2.3"  : "Neutral",  # mutual access obligation
    "2.4"  : "Pro",      # change orders require both parties to sign

    # Section 3 — Payment Terms
    "3.1"  : "Pro",      # 30-day payment terms
    "3.2"  : "Neutral",  # administrative
    "3.3"  : "Con",      # 18% interest — punitive
    "3.4"  : "Con",      # suspend immediately — unilateral
    "3.5"  : "Neutral",  # GST statutory clause

    # Section 4 — Intellectual Property
    "4.1"  : "Pro",      # deliverables vest in client on payment
    "4.2"  : "Pro",      # royalty-free licence granted
    "4.3"  : "Neutral",  # mutual background IP retention

    # Section 5 — Confidentiality
    "5.1"  : "Neutral",  # mutual NDA
    "5.2"  : "Neutral",  # need-to-know — standard
    "5.3"  : "Pro",      # fair carve-outs
    "5.4"  : "Neutral",  # 3-year survival

    # Section 6 — Warranties
    "6.1"  : "Pro",      # service provider warranty
    "6.2"  : "Pro",      # IP non-infringement warranty
    "6.3"  : "Con",      # ALL CAPS disclaimer of all implied warranties
    "6.4"  : "Neutral",  # client warranty — balanced

    # Section 7 — Limitation of Liability
    "7.1"  : "Con",      # excludes all indirect/consequential damages
    "7.2"  : "Con",      # hard cap at 3 months fees
    "7.3"  : "Pro",      # carve-out for fraud/gross negligence

    # Section 8 — Termination
    "8.1"  : "Neutral",  # standard duration
    "8.2"  : "Pro",      # either party, 60 days notice
    "8.3"  : "Pro",      # 30-day cure period before termination
    "8.4"  : "Con",      # client can terminate immediately without notice
    "8.5"  : "Pro",      # pro-rata payment + delivery of deliverables

    # Section 9 — Dispute Resolution
    "9.1"  : "Pro",      # good faith negotiation
    "9.2"  : "Pro",      # 30-day mediation window
    "9.3"  : "Pro",      # mutual arbitrator appointment
    "9.4"  : "Pro",      # equal cost sharing

    # Section 10 — Force Majeure
    "10.1" : "Pro",      # notice + mitigation required
    "10.2" : "Pro",      # either party can exit after 90 days

    # Section 11 — Indemnification
    "11.1" : "Pro",      # service provider indemnifies client
    "11.2" : "Neutral",  # reciprocal — balanced
    "11.3" : "Neutral",  # procedural conditions — balanced

    # Section 12 — Governing Law (hard-neutral)
    "12.1" : "Neutral",
    "12.2" : "Neutral",

    # Section 13 — General Provisions
    "13.1" : "Neutral",
    "13.2" : "Pro",      # amendments require BOTH parties to sign
    "13.3" : "Neutral",
    "13.4" : "Pro",      # assignment requires prior written consent
    "13.5" : "Neutral",
}

CRITICAL_CON = {"3.3", "3.4", "7.1"}     # must be flagged Con
HIGH_PRO     = {"4.1", "8.2", "8.3", "13.2"}
HARD_NEUTRAL = {"1", "12.1", "12.2"}

# ─────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python check_accuracy_service.py results.json")
    sys.exit(1)

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

results   = data.get("results", data)
predicted = {str(r["clause_id"]).strip(): r["label"].strip().capitalize() for r in results}

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

con_ids = [c for c, e in GROUND_TRUTH.items() if e == "Con"]
pro_ids = [c for c, e in GROUND_TRUTH.items() if e == "Pro"]
neu_ids = [c for c, e in GROUND_TRUTH.items() if e == "Neutral"]

con_correct  = sum(1 for c in con_ids  if predicted.get(c) == "Con")
pro_correct  = sum(1 for c in pro_ids  if predicted.get(c) == "Pro")
neu_correct  = sum(1 for c in neu_ids  if predicted.get(c) == "Neutral")
crit_correct = sum(1 for c in CRITICAL_CON if predicted.get(c) == "Con")
high_correct = sum(1 for c in HIGH_PRO     if predicted.get(c) == "Pro")
hard_correct = sum(1 for c in HARD_NEUTRAL if predicted.get(c) == "Neutral")

S  = "─" * 65
S2 = "═" * 65

print(f"\n{S2}")
print(f"  LEXANALYZE ACCURACY REPORT — SERVICE AGREEMENT")
print(f"{S2}\n")
print(f"  Overall accuracy   : {correct}/{scored} = {overall_pct}%")
print(f"  Clauses not found  : {not_found}\n")
print(f"  ── By label ──────────────────────────────────────────")
print(f"  Con     : {con_correct}/{len(con_ids)}  ({round(con_correct/len(con_ids)*100,1) if con_ids else 0}%)")
print(f"  Pro     : {pro_correct}/{len(pro_ids)}  ({round(pro_correct/len(pro_ids)*100,1) if pro_ids else 0}%)")
print(f"  Neutral : {neu_correct}/{len(neu_ids)}  ({round(neu_correct/len(neu_ids)*100,1) if neu_ids else 0}%)\n")
print(f"  ── Priority checks ───────────────────────────────────")
print(f"  Critical CON (3.3, 3.4, 7.1)     : {crit_correct}/3  {'✅ PASS' if crit_correct==3 else '❌ FAIL'}")
print(f"  High PRO    (4.1, 8.2, 8.3, 13.2): {high_correct}/4  {'✅ PASS' if high_correct>=3 else '⚠ PARTIAL' if high_correct>=2 else '❌ FAIL'}")
print(f"  Hard-neutral (1, 12.1, 12.2)      : {hard_correct}/3  {'✅ PASS' if hard_correct==3 else '❌ FAIL'}\n")

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
        tag = "⚠ CRITICAL" if cid in CRITICAL_CON else ("⚠ HIGH PRO" if cid in HIGH_PRO else "")
        print(f"  {cid:<8} expected={exp:<9} got={pred:<12} {tag}")
else:
    print("  Perfect score!")
print(f"\n{S2}\n")