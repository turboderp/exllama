#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup
import shutil
import pathlib

def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    lines = codecs.open(file_path, encoding="utf-8").readlines()
    requirements = []
    dependency_links = []
    for line in lines:
        if line.startswith("-f"):
            dependency_links.append(line.split("-f")[1])
        else:
            requirements.append(line)
    return requirements, dependency_links

requirements, dependency_links = read("requirements.txt")

print("Creating temporary package directory")

package_target_dir="exllama_lib"
extension_dir = "exllama_ext"

# Clean state before trying to package again
try:
    shutil.rmtree(package_target_dir)
except FileNotFoundError:
    pass

os.makedirs(package_target_dir)

exported_modules = [
        "cuda_ext",
        "lora",
        "generator",
        "model_init",
        "model",
        "perplexity",
        "tokenizer"
]

for exported_file in exported_modules:
    python_file = f"{exported_file}.py"
    shutil.copyfile(python_file, os.path.join(package_target_dir, python_file))

shutil.copytree(extension_dir, os.path.join(package_target_dir, extension_dir))
pathlib.Path(os.path.join(package_target_dir, "__init__.py")).touch(exist_ok=True)

def package_files(directory):
    # Recursive copy of packaged data dir
    # Originally from: https://stackoverflow.com/a/36693250/8628527
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files(os.path.join(package_target_dir, extension_dir))


setup(
    name="exllama_lib",
    version="0.1.0",
    author="turboderp",
    author_email="<not provided>",
    maintainer="turboderp",
    maintainer_email="<not provided>",
    license="MIT",
    url="https://github.com/turboderp/exllama",
    description="Memory efficient implementation of llama inference.",
    python_requires=">=3.8",
    install_requires=requirements,
    dependency_links=dependency_links,
    packages=[package_target_dir],
    package_data={
        package_target_dir: extra_files
    },
    py_modules=exported_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: Pytest",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ]
)