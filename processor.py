import sys
trace = []

trace_path = "C:\\Users\\Amadeus\\Documents\\Code\\ML_LIRS\\trace\\{0}".format(sys.argv[1])
print(trace_path)

with open(trace_path, "r") as f:
    for line in f.readlines():
        if not line == "*\n" and not line.strip() == "":
            temp = line.split(" ")
            trace.append(temp[0].strip())

max_lines = 100000
with open(trace_path, "w") as f:
    for i, line in enumerate(trace):
        f.write(line + "\n")
        if i == max_lines:
            break