import setuptools

setuptools.setup(
      name='dpu_utils',
      version='0.1.2',
      description='Python utilities used by Deep Program Understanding',
      url='https://deepproceduralintelligence.visualstudio.com/dpu-utils/',
      author='Deep Program Understanding',
      author_email='miallama@microsoft.com',
      packages=setuptools.find_packages(),
      install_requires=[
          'azure-storage', 'numpy'
      ],
      zip_safe=False)
