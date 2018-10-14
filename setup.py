import setuptools

setuptools.setup(
      name='dpu_utils',
      version='0.1.15',
      license='MIT',
      description='Python utilities used by Deep Procedural Intelligence',
      url='https://deepproceduralintelligence.visualstudio.com/dpu-utils/',
      author='Deep Procedural Intelligence',
      author_email='miallama@microsoft.com',
      packages=setuptools.find_packages(),
      install_requires=[
          'azure-storage', 'numpy'
      ],
      zip_safe=False)
