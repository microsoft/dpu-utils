# DPU Utilities

This contains a set of utilities used across projects of the [DPU team](https://www.microsoft.com/en-us/research/project/program/).

Generic Utilities:
* `dpu_utils.utils.RichPath` a convenient way of using both paths and Azure paths in your code.
* `dpu_utils.utils.*Iterator` iterator wrappers that can parallelize their iteration in other threads/processes.
* `dpu_utils.utils.{load,save}_json[l]_gz` convenience methods for loading .json[l].gz from the filesystem.
* `dpu_utils.utils.git_tag_run` that tags the current working directory git the state of the code.
* `dpu_utils.utils.run_and_debug` when an exception happens, start a debug session. Usually a wrapper of `__main__`.
* `dpu_utils.utils.ChunkWriter` that helps writing chunks to the output.

TensorFlow Utilities:
* `dpu_utils.tfutils.GradRatioLoggingOptimizer` a wrapper around optimizers that logs the ratios of grad norms to parameter norms.
* `dpu_utils.tfutils.unsorted_segment_logsumexp`
* `dpu_utils.tfutils.unsorted_segment_log_softmax`
* `dpu_utils.tfutils.TFVariableSaver` save TF variables in an object that can be pickled.

General Machine Learning Utilities:
* `dpu_utils.mlutils.CharTensorizer` for character-level tensorization.
* `dpu_utils.mlutils.Vocabulary` a str to int vocabulary for machine learning models

TensorFlow Models:
* `dpu_utils.tfmodels.SparseGGNN` a sparse GGNN implementation.
* `dpu_utils.tfmodels.AsyncGGNN` an asynchronous GGNN implementation.

Code-related Utilities
* `dpu_utils.codeutils.split_identifier_into_parts` split identifiers into subtokens on CamelCase and snake_case.
* `dpu_utils.codeutils.{Lattice, CSharpLattice}` represent lattices and some useful operations in Python.

## Tests

### Run the unit tests

```bash
python setup.py test
```

### Generate code coverage reports

```bash
# pip install coverage
coverage run --source dpu_utils/ setup.py test && \
  coverage html
```

The resulting HTML file will be in `htmlcov/index.html`.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
