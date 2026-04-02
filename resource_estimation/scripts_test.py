import subprocess


def test_clifford_t():
    result = subprocess.run(["python", "scripts/clifford_t.py"])
    assert result


def test_scaling():
    result = subprocess.run(["python", "scripts/scaling.py", "10", "20"])
    assert result


def test_rz_games():
    result = subprocess.run(["python", "scripts/rz_games.py", ".122441", "12", "0"])
    assert result
