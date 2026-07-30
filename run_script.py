#!/usr/bin/env python
import sys
import os
import traceback

# Redirect output to file
logfile = open("c:\\Users\\User\\Downloads\\debug.log", "w")
sys.stdout = logfile
sys.stderr = logfile

try:
    os.chdir("c:\\Users\\User\\Downloads")
    print(f"Working directory: {os.getcwd()}", flush=True)
    with open("Code for Model Building.py") as f:
        code = f.read()
    print("Script loaded. Starting execution...", flush=True)
    exec(code, {"__name__": "__main__"})
    print("Script completed successfully!", flush=True)
except Exception as e:
    print("ERROR:", str(e), file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
finally:
    logfile.close()
