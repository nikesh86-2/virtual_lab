#!/usr/bin/env python3
import sys, os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if len(sys.argv) < 2:
    eprint("usage: trafl_finds_lowestE_frame.py file.trafl")
    sys.exit(1)

inpfilename = sys.argv[1]
if not inpfilename.endswith(".trafl"):
    eprint(f"extension of input file: {inpfilename} has to be .trafl")
    sys.exit(1)

if not os.path.exists(inpfilename):
    eprint(f"specified file: {inpfilename} doesn't exist")
    sys.exit(1)

counter = 1
lowest_frame = 1
lowest_energy = 1e9
lowestE_header = ""
lowestE_coords = ""
with open(inpfilename) as inpfile:
    for line in inpfile:
        if len(line) < 100:
            parts = line.split()
            try:
                curr_energy = float(parts[-2])
            except Exception:
                counter += 1
                continue
            if lowest_energy > curr_energy:
                lowest_energy = curr_energy
                lowest_frame = counter
                lowestE_header = line.rstrip("\n")
                lowestE_coords = next(inpfile, "").rstrip("\n")
            counter += 1

print(lowest_frame, lowest_energy)
outfilename = inpfilename.replace(".trafl", "_minE.trafl")
print(outfilename)
with open(outfilename, "w") as outfile:
    print(lowestE_header, file=outfile)
    print(lowestE_coords, file=outfile)
