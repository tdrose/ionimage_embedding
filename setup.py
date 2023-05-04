from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='deepims_clust',
    version='0.1.0',

    description='Deep learning-based clustering approach to cluster ion images from mass spectrometry imaging data '
                'across datasets. Optimized to work with datasets from the METASPACE database.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/tdrose/deep_MultiData_mzClustering',
    project_urls={  # Optional
        'Source': 'https://github.com/tdrose/deep_MultiData_mzClustering',
        #'Publication': ""
    },

    author="Tim Daniel Rose",
    author_email="tim.rose@embl.de",

    license='MIT',

    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'metaspace2020',
        'scikit-learn'
    ],
    python_requires=">=3.8",

    zip_safe=False,


    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3"
    ]
)
