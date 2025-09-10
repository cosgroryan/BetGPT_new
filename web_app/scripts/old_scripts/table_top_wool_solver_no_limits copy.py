import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Contract data
contracts = [
    {"name": "15.5 - 15.9", "min": 15.5, "max": 15.9, "price": 3900},
    {"name": "16.0 - 16.4", "min": 16.0, "max": 16.4, "price": 3480},
    {"name": "16.5 - 16.9", "min": 16.5, "max": 16.9, "price": 3180},
    {"name": "17.0 - 17.4", "min": 17.0, "max": 17.4, "price": 2880},
    {"name": "17.5 - 17.9", "min": 17.5, "max": 17.9, "price": 2640},
    {"name": "18+", "min": 18.0, "max": 25.0, "price": 1956},  # upper bound set high
]

# Line data
lines = [
    {"line": 7, "bales": 2, "micron": 14.66380952},
    {"line": 1, "bales": 15, "micron": 15.75960347},
    {"line": 2, "bales": 13, "micron": 16.43170029},
    {"line": 3, "bales": 13, "micron": 16.91251908},
    {"line": 4, "bales": 10, "micron": 17.36435453},
    {"line": 5, "bales": 10, "micron": 17.91386555},
    {"line": 6, "bales": 9, "micron": 18.88604651},
]

# Create LP problem
model = LpProblem(name="wool_allocation", sense=LpMaximize)

# Decision variables: bales from each line to each contract
x = {}
for i, line in enumerate(lines):
    for j, contract in enumerate(contracts):
        x[(i, j)] = LpVariable(
            name=f"x_{line['line']}_{j}", lowBound=0, upBound=line["bales"], cat="Integer"
        )

# Objective: maximise revenue
model += lpSum(
    x[(i, j)] * contracts[j]["price"]
    for i in range(len(lines))
    for j in range(len(contracts))
)

# Bale limits per line
for i, line in enumerate(lines):
    model += lpSum(x[(i, j)] for j in range(len(contracts))) <= line["bales"]

# Micron constraints per contract
for j, contract in enumerate(contracts):
    total_bales = lpSum(x[(i, j)] for i in range(len(lines)))
    model += (
        lpSum(x[(i, j)] * lines[i]["micron"] for i in range(len(lines)))
        >= contract["min"] * total_bales
    )
    model += (
        lpSum(x[(i, j)] * lines[i]["micron"] for i in range(len(lines)))
        <= contract["max"] * total_bales
    )

# Solve
model.solve()

# Extract results
allocation = []
for i, line in enumerate(lines):
    for j, contract in enumerate(contracts):
        bales_alloc = int(x[(i, j)].value())
        if bales_alloc > 0:
            allocation.append({
                "Line": line["line"],
                "Contract": contracts[j]["name"],
                "Bales": bales_alloc,
                "Revenue": bales_alloc * contracts[j]["price"]
            })

df_allocation = pd.DataFrame(allocation)
total_revenue = df_allocation["Revenue"].sum()

# Save to CSV
df_allocation.to_csv("wool_allocation.csv", index=False)

print(f"✅ Allocation saved to wool_allocation.csv")
print(f"💰 Total Revenue: ${total_revenue:,.2f}")
