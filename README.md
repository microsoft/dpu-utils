# DPU Utilities
[![Build Status](https://deepproceduralintelligence.visualstudio.com/dpu-utils/_apis/build/status/Microsoft.dpu-utils?branchName=master)](https://deepproceduralintelligence.visualstudio.com/dpu-utils/_build/latest?definitionId=3)

This contains a set of utilities used across projects of the [DPU team](https://www.microsoft.com/en-us/research/project/program/).

## Python

Stored in the `python` subdirectory, published as the `dpu-utils` package.

### Installation

```bash
pip install dpu-utils
```

### Overview
Below you can find an overview of the utilities included. Detailed documentation
is provided at the docstring of each class.

##### Generic Utilities
* `dpu_utils.utils.ChunkWriter` provides a convenient API for writing output in multiple parts (chunks).
* `dpu_utils.utils.RichPath` an API that abstract local and Azure Blob paths in your code.
* `dpu_utils.utils.*Iterator` Wrappers that can parallelize and shuffle iterators.
* `dpu_utils.utils.{load,save}_json[l]_gz` convenience API for loading and writing `.json[l].gz` files.
* `dpu_utils.utils.git_tag_run` tags the current working directory git the state of the code.
* `dpu_utils.utils.run_and_debug` when an exception happens, start a debug session. Usually a wrapper of `__main__`.

##### General Machine Learning Utilities
* `dpu_utils.mlutils.Vocabulary` map elements into unique integer ids and back.
    Commonly used in machine learning models that work over discrete data (e.g. 
    words in NLP). Contains methods for converting an list of tokens into their
    "tensorized" for of integer ids.  
* `dpu_utils.mlutils.BpeVocabulary` a vocabulary for machine learning models that employs BPE (via `sentencepiece`).
* `dpu_utils.mlutils.CharTensorizer` convert character sequences into into tensors, commonly used
    in machine learning models whose input is a list of characters.

##### Code-related Utilities
* `dpu_utils.codeutils.split_identifier_into_parts()` split identifiers into subtokens on CamelCase and snake_case.
* `dpu_utils.codeutils.{Lattice, CSharpLattice}` represent lattices and useful operations on lattices in Python.
* `dpu_utils.codeutils.get_language_keywords()` an API to retrieve the keyword tokens for many programming languages.
* `dpu_utils.codeutils.deduplication.DuplicateDetector` API to detects (near)duplicates in codebases.

##### TensorFlow 1.x Utilities
* `dpu_utils.tfutils.get_activation` retrieve activations function by name.
* `dpu_utils.tfutils.GradRatioLoggingOptimizer` a wrapper around optimizers that logs the ratios of grad norms to parameter norms.
* `dpu_utils.tfutils.TFVariableSaver` save TF variables in an object that can be pickled.

Unsorted segment operations following TensorFlow's [`unsorted_segment_sum`](https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum) operations:
* `dpu_utils.tfutils.unsorted_segment_logsumexp`
* `dpu_utils.tfutils.unsorted_segment_log_softmax`
* `dpu_utils.tfutils.unsorted_segment_softmax`

##### TensorFlow 2.x Utilities
* `dpu_utils.tf2utils.get_activation_function_by_name` retrieve activation functions by name.
* `dpu_utils.tf2utils.gelu` The GeLU activation function.
* `dpu_utils.tf2utils.MLP` An MLP layer.

Unsorted segment operations following TensorFlow's [`unsorted_segment_sum`](https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum) operations:
* `dpu_utils.tf2utils.unsorted_segment_logsumexp` 
* `dpu_utils.tf2utils.unsorted_segment_log_softmax`
* `dpu_utils.tf2utils.unsorted_segment_softmax`


##### TensorFlow Models:
* `dpu_utils.tfmodels.SparseGGNN` a sparse GGNN implementation.
* `dpu_utils.tfmodels.AsyncGGNN` an asynchronous GGNN implementation.

These models have not been tested with TF 2.0.

##### PyTorch Utilities
* `dpu_utils.ptutils.BaseComponent` a wrapper abstract class around `nn.Module` that 
   takes care of essential elements of most neural network components.
* `dpu_utils.ptutils.ComponentTrainer` a training loop for `BaseComponent`s.


### Command-line tools

#### Approximate Duplicate Code Detection
You can use the `deduplicationcli` command to detect duplicates in pre-processed source code, by invoking
```bash
deduplicationcli DATA_PATH OUT_JSON
```
where `DATA_PATH` is a file containing tokenized `.jsonl.gz` files and `OUT_JSON` is the target output file.
For more options look at `--help`.

An exact (but usually slower) version of this can be found [here](https://github.com/Microsoft/near-duplicate-code-detector)
along with code to tokenize Java, C#, Python and JavaScript into the relevant formats.

### Tests

#### Run the unit tests

```bash
python setup.py test
```

#### Generate code coverage reports

```bash
# pip install coverage
coverage run --source dpu_utils/ setup.py test && \
  coverage html
```

The resulting HTML file will be in `htmlcov/index.html`.

## .NET

Stored in the `dotnet` subdirectory.

Generic Utilities:
* `Microsoft.Research.DPU.Utils.RichPath`: a convenient way of using both paths and Azure paths in your code.

Code-related Utilities:
* `Microsoft.Research.DPU.CSharpSourceGraphExtraction`: infrastructure to extract Program Graphs from C# projects.

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
