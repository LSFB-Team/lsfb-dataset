import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lsfb-dataset",
    version="0.1.0",
    author="Jérôme Fink",
    author_email="jerome.fink@unamur.be",
    description="A companion library for the LSFB-dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jefidev/lsfb-dataset",
    project_urls={
        "Bug Tracker": "https://github.com/Jefidev/lsfb-dataset/issues",
        "Documentation": "https://jefidev.github.io/lsfb-dataset/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "Pillow",
        "opencv-python",
        "mediapipe",
        "tqdm",
    ],
)
