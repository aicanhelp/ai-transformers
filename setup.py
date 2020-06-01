from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    metadata_version='2.1',
    name='ai-transformersx',
    version='0.4.0',
    description='tools for using huggingface/transformers more easily',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aicanhelp/ai-transformers',
    author='modongsong',
    author_email='modongsongml@163.com',
    license='MIT',

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    keywords='deeplearning tools development',  # Optionalz
    packages=find_packages(),  # Required
    python_requires='>=3.5, <4',
    install_requires=[],  # Optional

    extras_require={  # Optional
        'dev': ['check-manifest'],
        'test': ['coverage', 'pytest'],
    },

    # package_data={  # Optional
    #     'sample': ['package_data.dat'],
    # },

    # data_files=[('my_data', ['data/data_file'])],  # Optional

    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/aicanhelp/ai-transformersx/issues',
        'Source': 'https://github.com/aicanhelp/ai-transformers/',
    },
)
