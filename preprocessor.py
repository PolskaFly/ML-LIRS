trace = []

trace_path = "/Users/polskafly/Desktop/REU/trace/Financial1"
file = open("trace/Financial1", "w")
with open(trace_path, "r") as f:
    for line in f.readlines():
        if not line == "*\n" and not line.strip() == "":
            temp = line.split(",")
            file.write(temp[1] + "\n");

