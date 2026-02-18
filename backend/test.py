import inspect

import importlib
import pipeline

import sys
import os

sys.path.append(os.path.dirname(__file__))


importlib.reload(pipeline)
print(inspect.signature(pipeline.run_pipeline))

print(pipeline.__file__)
from pipeline import run_pipeline



# ===== 10 TEST CLAUSES =====
test_clauses = [
    {
        "clause_id": 1,
        "section": "Termination",
        "text": "The company reserves the right to terminate employment at any time without notice.",
        "position": 1
    },
    {
        "clause_id": 2,
        "section": "Compensation",
        "text": "The employee shall receive compensation and benefits as agreed in writing.",
        "position": 2
    },
    {
        "clause_id": 3,
        "section": "Definitions",
        "text": "For the purpose of this agreement, the term employee refers to the hired individual.",
        "position": 3
    },
    {
        "clause_id": 4,
        "section": "Termination",
        "text": "Either party may terminate this agreement with thirty (30) days prior written notice.",
        "position": 4
    },
    {
        "clause_id": 5,
        "section": "Conduct",
        "text": "Failure to comply with company policies may result in immediate suspension.",
        "position": 5
    },
    {
        "clause_id": 6,
        "section": "Leave",
        "text": "Employees are entitled to annual paid leave according to company policy.",
        "position": 6
    },
    {
        "clause_id": 7,
        "section": "Modification",
        "text": "The employer may modify terms of employment at its sole discretion.",
        "position": 7
    },
    {
        "clause_id": 8,
        "section": "Severance",
        "text": "In case of termination, severance compensation shall be provided.",
        "position": 8
    },
    {
        "clause_id": 9,
        "section": "Governing Law",
        "text": "This agreement shall be governed by the laws of the jurisdiction.",
        "position": 9
    },
    {
        "clause_id": 10,
        "section": "Termination",
        "text": "The employer reserves the right to dismiss employees immediately for misconduct.",
        "position": 10
    }
]

# ===== RUN PIPELINE =====
results = run_pipeline(test_clauses,role="Employee")

# ===== PRINT RESULTS =====
print("\n=== PIPELINE OUTPUT ===\n")
for r in results:
    print(f"Clause ID: {r['clause_id']}")
    print(f"Section  : {r['section']}")
    print(f"Label    : {r['label']}")
    print(f"Confidence: {r['confidence']:.3f}")
    print(f"Original : {r['original']}")
    print(f"Amended  : {r['amended']}")
    print("-" * 60)