"""Package the tensor-only closed-path shaping environment."""
from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
dependencies = [line.strip() for line in (root / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.strip().startswith(("#", "-"))]
setup(
    name="closed-path-shaping",
    version="0.1.0",
    description="Persistent multi-UAV coverage through closed-path shaping with TD-MPC2",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["mani_skill*"]),
    python_requires=">=3.9",
    install_requires=dependencies,
)
