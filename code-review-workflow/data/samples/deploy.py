import subprocess


def run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def calc(expr: str):
    return eval(expr)
