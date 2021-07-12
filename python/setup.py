import setuptools, os

if os.path.exists('../README.md'):
    with open('../README.md') as f:
        long_description = f.read()
else:
    long_description = ""

setuptools.setup(
      name='dpu_utils',
      version='0.5.0',
      license='MIT',
      description='Python utilities used by Deep Procedural Intelligence',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/microsoft/dpu-utils',
      author='Deep Procedural Intelligence',
      packages=setuptools.find_packages(),
      python_requires=">=3.6.1",
      include_package_data=True,
      install_requires=[
          'azure-storage-blob', 'azure-identity', 'numpy', 'docopt', 'tqdm', 'SetSimilaritySearch', 'sentencepiece', 'cffi', 'regex'
      ],
      scripts=['dpu_utils/codeutils/deduplication/deduplicationcli'],
      test_suite="tests",
      zip_safe=False)
