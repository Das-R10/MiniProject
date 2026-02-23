from clause_parser import split_into_clauses

# Read sample file
with open("sample_contract.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Run parser
clauses = split_into_clauses(text)

# Print results
print(f"\nTotal Clauses Found: {len(clauses)}\n")

for clause in clauses:
    print("Clause ID:", clause["clause_id"])
    print("Text:", clause["text"][:120], "...")
    print("Position:", clause["position"])
    print("-" * 50)