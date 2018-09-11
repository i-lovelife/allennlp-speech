from setuptools import setup, find_packages

setup(name="allennlp-speech",
    version="0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=[
        'allennlp==0.6.1',
        'scipy==1.1.0',
        'jupyter',
        'jupyterlab',
        'mypy',
        'librosa==0.6.2'
    ],
    python_requires='>=3.6.1'
)
