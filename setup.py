#!/usr/bin/env python3
"""Setup script for ADAN Trading Bot"""

from setuptools import setup, find_packages

setup(
    name="adan_trading_bot",
    version="0.1.0",
    description="ADAN Trading Bot - Advanced Deep Adaptive Networks for Algorithmic Trading",
    author="ADAN Team",
    author_email="info@adan.trading",
    url="https://github.com/Cabrel10/ADAN0",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "yfinance>=0.2.0",
        "pandas-ta>=0.3.14b0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "optuna>=3.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "TA-Lib>=0.4.28",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
        ],
        "colab": [
            "google-colab>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
    zip_safe=False,
)
