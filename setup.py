from setuptools import setup, find_packages

setup(
    name="kiri_ocr",
    version="0.1.0",
    description="A lightweight OCR library for Khmer and English documents",
    author="Blizzer",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "opencv-python",
        "Pillow",
        "tqdm",
        "PyYAML",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    scripts=['scripts/kiri-ocr'],
)
