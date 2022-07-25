from setuptools import setup, find_packages

setup(
    name='anzu',
    version='1.0',
    packages=find_packages(),
    package_dir={'anzu' : 'anzu',
                 'fields' :'fields'},
    scripts=['bin/run_fields.py',
             'bin/zcv.py'],
    package_data={'anzu': ['data/*']},
    long_description=open('README.md').read(),
    )
