from setuptools import setup

setup(name='dpu_utils',
      version='0.1',
      description='Python utilities used by Deep Program Understanding',
      url='https://deepproceduralintelligence.visualstudio.com/dpu-utils/',
      author='Deep Program Understanding',
      author_email='miallama@microsoft.com',
      packages=['dpu_utils'],
      install_requires=[
          'azure', 'numpy'
      ],
      zip_safe=False)