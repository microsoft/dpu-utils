DPU Utilities
====

This contains a set of utilities used across projects.

Generic Utilities:
* `dpu_utils.utils.RichPath` a convinient way of using both paths and Azure paths in your code.
* `dpu_utils.utils.Vocabulary` a str to int vocabulary for machine learning models
* `dpu_utils.utils.*Iterator` iterator wrappers that can parallelize their iteration in other threads/processes.
* `dpu_utils.utils.{load,save}_json_gz` convinience methods for loading .json.gz from the filesystem.

TensorFlow Utilities:
* `dpu_utils.tfutils.GradRatioLoggingOptimizer` a wrapper around optimizers that logs the ratios of grad norms to parameter norms.

Code-related Utilities
* `dpu_utils.codeutils.split_identifier_into_parts` split identifiers into subtokens on CamelCase and snake_case
* `dpu_utils.codeutils.{Lattice, CSharpLattice}` represent lattices and some useful operations in Python


Use
=======
First install `dpu-utils` in your environment as

```
pip install --extra-index-url https://dpucode.z6.web.core.windows.net/simple dpu-utils
```
Then use the code.

If you don't have access, it may be that (a) you are not in CorpNet (b) your IP is not whitelisted. Contact DPU for this.


Deploying updates to packages to private PyPi
=======
Once a new version is available:
* Update the version number in `setup.py`
* Add the new version in `packagesToUpload.txt`
* Commit and run the Build in VSTS.
* Download the artifact `.zip` and unzip locally.
* Upload it to the `dpucode/$web` container.

