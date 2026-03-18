from setuptools import setup, find_packages
setup(
    name="poker_ai_v4", version="4.0.0",
    packages=find_packages(),
    install_requires=["rlcard>=1.0.5","torch>=2.0.0","numpy>=1.24.0","tensorboard>=2.13.0"],
    python_requires=">=3.10",
)
