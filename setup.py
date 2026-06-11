import setuptools


setuptools.setup(
    name='rimo_color_match',
    version='1.1.0',
    author='RimoChan',
    author_email='the@librian.net',
    description='rimo_color_match',
    long_description=open('readme.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RimoChan/rimo_color_match',
    packages=['rimo_color_match'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
