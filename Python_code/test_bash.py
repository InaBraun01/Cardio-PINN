
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a", type=float, default=5.0, help="Value for a")
parser.add_argument("--b", type=float, default=5.0, help="Value for b")
args = parser.parse_args()

a = args.a
b = args.b
print(f"a is {a}")
print(f"b is {b}")
print(f"a+b is {a+b}")