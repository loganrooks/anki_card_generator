from setuptools import setup, find_packages

setup(
    name='anki_card_generator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'openai',
        'ebooklib',
        'beautifulsoup4',
    ],
    entry_points={
        'console_scripts': [
            'anki_card_generator=anki_card_generator:main',
        ],
    },
    author='Logan Rooks',
    author_email='logansrooks@gmail.com',
    description='gi',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/loganrooks/anki_card_generator',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
