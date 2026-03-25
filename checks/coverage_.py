#!/usr/bin/env python3

import argparse
import subprocess
import sys
from glob import glob

# Slow test files and their corresponding source files
# SLOW_FILES = {
#     "tests/test_stim_functions.py": "resource_estimation/stim_functions.py",
#     "tests/test_cliff_rz.py": "resource_estimation/cliff_rz.py",
#     "tests/test_reversible_circuits.py": "resource_estimation/reversible_circuits.py",
# }


# def get_test_files_and_omit_sources(fast: bool):
#     all_tests = glob("resource_estimation/*_test.py")
#     omit_sources = [
#         "scripts/circuits.py",
#     ]
#     if fast:
#         for test_file, src_file in SLOW_FILES.items():
#             if test_file in all_tests:
#                 all_tests.remove(test_file)
#             omit_sources.append(src_file)

#     return all_tests, omit_sources


def main():
    parser = argparse.ArgumentParser(description="Run coverage tests with optional fast mode")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests and their coverage")
    args = parser.parse_args()

    # test_files, omit_sources = get_test_files_and_omit_sources(fast=args.fast)
    omit_sources = ["scripts/*.py"]
    test_files = glob("resource_estimation/*_test.py")
    if not test_files:
        print("❌ No test files found to run.")
        sys.exit(1)

    print(f"Running coverage on {len(test_files)} test file(s)...")

    try:
        run_cmd = ["coverage", "run"]
        # if omit_sources:
        #     run_cmd += ["--omit", ",".join(omit_sources)]
        run_cmd += ["-m", "pytest", *test_files]

        subprocess.run(run_cmd, check=True)

    except subprocess.CalledProcessError:
        print("❌ Tests failed.")
        sys.exit(1)

    try:
        subprocess.run(["coverage", "report", "-m", "--fail-under=100"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Coverage below 100%")
        sys.exit(1)


if __name__ == "__main__":
    main()
