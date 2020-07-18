"""
Fashionpedia is a new dataset which consists of two parts:  (1) an ontology built by fashion experts containing 27 main apparel categories, 19 apparel parts, 294 fine-grained attributes and their relationships; (2) a dataset with 48k everyday and celebrity event fashion images annotated with segmentation masks and their associated per-mask fine-grained attributes, built upon the Fashionpedia ontology. Fashionpedia API enables reading, and visualizing annotations, and evaluating results.
"""
DOCLINES = (__doc__ or '')

import os.path
import setuptools
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fashionpedia"))


with open("requirements.txt") as f:
    reqs = f.read()

DISTNAME = "fashionpedia"
DESCRIPTION = "Python API for Fashionpedia dataset."
AUTHOR = "Menglin Jia"
REQUIREMENTS = (reqs.strip().split("\n"),)


if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        version="1.1",
        author=AUTHOR,
        author_email="mj493@cornell.edu",
        description=DESCRIPTION,
        long_description=DOCLINES,
        long_description_content_type='text/markdown',
        url="https://github.com/KMnP/fashionpedia-api",
        packages=setuptools.find_packages(),
        python_requires='>=3.6',
        install_requires=REQUIREMENTS,
    )
