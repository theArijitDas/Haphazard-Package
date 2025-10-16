from setuptools import setup, find_packages
from pathlib import Path
import re

# Get version from haphazard/__init__.py
init_path = Path(__file__).parent / "haphazard" / "__init__.py"
init_contents = init_path.read_text()
version_match = re.search(r'^__version__ = ["\']([^"\']+)["\']', init_contents, re.M)
if not version_match:
    raise RuntimeError("Cannot find version string in haphazard/__init__.py")
version = version_match.group(1)

setup(
    name="haphazard",
    version=version,
    author="Arijit Das",
    author_email="dasarijitjnv@gmail.com",
    description="A modular framework for registering and running haphazard datasets and models.",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/theArijitDas/Haphazard-Package/",
    project_urls={
        "Bug Tracker": "https://github.com/theArijitDas/Haphazard-Package/issues",
        "Source Code": "https://github.com/theArijitDas/Haphazard-Package/",
    },
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "scikit-learn",
        "torch",
        "statsmodels",
    ],
    extras_require={
        "orf3v": ["tdigest"],
        "all": ["tdigest"],
    },
    python_requires=">=3.10",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning haphazard models datasets registration framework",
)
