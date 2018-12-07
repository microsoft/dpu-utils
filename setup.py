import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
      name='dpu_utils',
      version='0.1.24',
      license='MIT',
      description='Python utilities used by Deep Procedural Intelligence',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/microsoft/dpu-utils',
      author='Deep Procedural Intelligence',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=[
          'azure-storage', 'numpy', 'docopt', 'tqdm', 'SetSimilaritySearch'
      ],
      scripts=['dpu_utils/codeutils/deduplication/deduplicationcli'],
      test_suite="tests",
      zip_safe=False)
