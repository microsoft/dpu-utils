import setuptools

setuptools.setup(
      name='dpu_utils',
      version='0.1.16',
      license='MIT',
      description='Python utilities used by Deep Procedural Intelligence',
      url='https://github.com/microsoft/dpu-utils',
      author='Deep Procedural Intelligence',
      packages=setuptools.find_packages(),
      install_requires=[
          'azure-storage', 'numpy'
      ],
      zip_safe=False)
