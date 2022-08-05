from setuptools import setup, find_packages

setup(
    name='Xenakis',
    extras_requires=dict(tests=['pytest']),
    packages=find_packages(where='src'),
    package_dir={"": "src"},
)