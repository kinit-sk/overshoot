from setuptools import setup, find_packages

setup(
    name="overshoot",
    version="0.1.0",
    description="Overshoot version of SGD and AdamW optimizers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jakub Kopal",
    author_email="jakub.kopal@kinit.sk",
    url="https://github.com/kinit-sk/overshoot",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)