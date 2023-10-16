from setuptools import setup, find_packages

# Read the README for long_description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line]

setup(
    name="trinary_tree",
    version="0.1.1",
    author="Henning Zakrisson",
    author_email="henning.zakrisson@gmail.com",
    description="Python package for the Trinary Tree algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henningz/trinary-tree",  # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
