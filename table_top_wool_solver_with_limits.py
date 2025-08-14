# pip install pulp pandas

import pandas as pd
from math import isfinite
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

# ----------------------------
# Settings
# ----------------------------
TOLERANCE = 1.5  # microns allowed below contract min for individual bale

# ----------------------------
# Inputs
# ----------------------------
contracts = [
    {"name": "15.5 - 15.9", "min": 15.5, "max": 15.9, "price": 3900},
    {"name": "16.0 - 16.4", "min": 16.0, "max": 16.4, "price": 3480},
    {"name": "16.5 - 16.9", "min": 16.5, "max": 16.9, "price": 3180},
    {"name": "17.0 - 17.4", "min": 17.0, "max": 17.4, "price": 2880},
    {"name": "17.5 - 17.9", "min": 17.5, "max": 17.9, "price": 2640},
    {"name": "18+",         "min": 18.0, "max": float("inf"), "price": 1956},
]

lines = [
    {"line": 7, "bales": 2,  "micron": 14.66380952},
    {"line": 1, "bales": 15, "micron": 15.75960347},
    {"line": 2, "bales": 13, "micron": 16.43170029},
    {"line": 3, "bales": 13, "micron": 16.89251908},
    {"line": 4, "bales": 10, "micron": 17.36435453},
    {"line": 5, "bales": 10, "micron": 17.89386555},
    {"line": 6, "bales": 9,  "micron": 18.88604651},
]

# ----------------------------
# Model
# ----------------------------
m = LpProblem("wool_allocation_with_tolerance", LpMaximize)

# Decision vars with per-bale eligibility based on TOLERANCE
x = {}
for i, line in enumerate(lines):
    for j, con in enumerate(contracts):
        eligible = (line["micron"] >= con["min"] - TOLERANCE) and (line["micron"] <= con["max"])
        ub = line["bales"] if eligible else 0
        x[(i, j)] = LpVariable(f"x_line{line['line']}_con{j}", lowBound=0, upBound=ub, cat="Integer")

# Objective: maximise revenue ($ per bale)
m += lpSum(x[(i, j)] * contracts[j]["price"] for i in range(len(lines)) for j in range(len(contracts)))

# Bale availability per line
for i, line in enumerate(lines):
    m += lpSum(x[(i, j)] for j in range(len(contracts))) <= line["bales"]

# Contract average micron constraints
for j, con in enumerate(contracts):
    total = lpSum(x[(i, j)] for i in range(len(lines)))
    m += lpSum(x[(i, j)] * lines[i]["micron"] for i in range(len(lines))) >= con["min"] * total
    if isfinite(con["max"]):
        m += lpSum(x[(i, j)] * lines[i]["micron"] for i in range(len(lines))) <= con["max"] * total

# Solve
m.solve()

# ----------------------------
# Outputs
# ----------------------------
alloc_rows = []
for i, line in enumerate(lines):
    for j, con in enumerate(contracts):
        v = int(x[(i, j)].value())
        if v > 0:
            alloc_rows.append({
                "Line": line["line"],
                "Line_avg_micron": round(line["micron"], 4),
                "Contract": contracts[j]["name"],
                "Contract_min": contracts[j]["min"],
                "Contract_max": (None if not isfinite(contracts[j]["max"]) else contracts[j]["max"]),
                "Bales": v,
                "Price_per_bale": contracts[j]["price"],
                "Revenue": v * contracts[j]["price"],
            })

df_alloc = pd.DataFrame(alloc_rows).sort_values(["Contract", "Line"]).reset_index(drop=True)
df_alloc.to_csv("wool_allocation_tolerance.csv", index=False)

# Contract summary with realised averages
summary = []
for j, con in enumerate(contracts):
    bales = sum(int(x[(i, j)].value()) for i in range(len(lines)))
    if bales > 0:
        mic_total = sum(int(x[(i, j)].value()) * lines[i]["micron"] for i in range(len(lines)))
        avg_mic = mic_total / bales
        revenue = sum(int(x[(i, j)].value()) * con["price"] for i in range(len(lines)))
        summary.append({
            "Contract": con["name"],
            "Min": con["min"],
            "Max": (None if not isfinite(con["max"]) else con["max"]),
            "Bales": bales,
            "Avg_micron": round(avg_mic, 4),
            "Revenue": revenue,
        })

df_summary = pd.DataFrame(summary).sort_values("Contract").reset_index(drop=True)
df_summary.to_csv("wool_allocation_contract_summary.csv", index=False)

print("Tolerance used:", TOLERANCE)
print("Status:", LpStatus[m.status])
print("Saved: wool_allocation_tolerance.csv and wool_allocation_contract_summary.csv")
print("Total revenue: ${:,.2f}".format(df_alloc["Revenue"].sum()))
