from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.readlines()

setup(
    name="egl",
    version="1.0.0",
    packages=find_packages(),
    install_requires=install_requires,
    description="A package of black-box optimization (BBO) algorithms.",
    author="Yedidya Kfir",
    author_email="yedidyakfir@gmail.com",
)
