from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns a list of requirements from the requirements.txt file.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newlines and whitespace
        requirements = [req.replace("\n", "") for req in requirements]

        # Ignore '-e .' if it's in the requirements.txt
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Amit Gupta',
    author_email='amitguptacse2028@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)