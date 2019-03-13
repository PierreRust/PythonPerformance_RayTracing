from setuptools import setup, find_packages
from Cython.Build import cythonize

with open('README.md', 'r') as f:
    README = f.read()


# Basic dependencies, required to run pyDCOP:
deps = [
        'pillow',
        'pyyaml',
        'cython'
    ]

# Extra dependencies, used to run tests
test_deps = [
    'coverage',
    'pytest',
    'mypy'
]

# Extra dependencies, used to generate docs
doc_deps = [
    'sphinx',
    'sphinx_rtd_theme'
]


setup(
    name='raytracer',
    version='0.0.2',
    description='A simple ray tracer',
    long_description=README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers"
        "License :: OSI Approved :: Apache Software License",

        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",

        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    author='Pierre Rust',
    author_email='pierre.rust@orange.com',

    ext_modules = cythonize("raytracer/ray_cythonized.pyx"),

    keywords=['computer graphics, profiling, benchmarks'],
    install_requires=deps,
    tests_require=test_deps,

    scripts=[
        'raytracer/ray_tracer.py'
    ],
    packages=find_packages()
)
