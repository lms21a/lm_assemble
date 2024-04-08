from setuptools import setup, find_namespace_packages

def read_requirements():
    with open("requirements.txt") as f:
        return f.readlines()

if __name__ == '__main__':
    setup(
        name='lm_assemble',
        version='0.1',
        description='Assemble language models',
        packages=find_namespace_packages(),
        install_requires=read_requirements(),
        author='Lukas Slifierz',
    )