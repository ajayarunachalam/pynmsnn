import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pynmsnn',
    version='0.0.3',
    scripts=['pynmsnn_package'],
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "seaborn",
        "xgboost",
        "scipy",
        "tqdm",
        "scikit_learn",
        "matplotlib",
        "plot-metric",
        "regressormetricgraphplot",
    ],
    author="Ajay Arunachalam",
    author_email="ajay.arunachalam08@gmail.com",
    description="NeuroMorphic Predictive Model with Spiking Neural Networks (SNN) using Pytorch",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/ajayarunachalam/pynmsnn/',
    packages=setuptools.find_packages(),
    py_modules=['pyNM/spiking_binary_classifier', 'pyNM/nonspiking_binary_classifier', 'pyNM/spiking_multiclass_classifier', 'pyNM/nonspiking_multiclass_classifier','pyNM/spiking_regressor', 'pyNM/nonspiking_regressor', 'pyNM/cf_matrix'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
