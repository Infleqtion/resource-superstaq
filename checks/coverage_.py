#!/usr/bin/env python3
import checks_superstaq as checks
import sys

if __name__ == "__main__":
    sys.exit(checks.coverage_.run(*sys.argv[1:]))
