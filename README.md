# DPU Utilities ![PyPI - Python Version](https://img.shields.io/pypi/v/dpu-utils)![Anaconda](https://anaconda.org/conda-forge/dpu-utils/badges/version.svg)
[![Build Status](https://deepproceduralintelligence.visualstudio.com/dpu-utils/_apis/build/status/Microsoft.dpu-utils?branchName=master)](https://deepproceduralintelligence.visualstudio.com/dpu-utils/_build/latest?definitionId=3)


This contains a set of utilities used across projects of the [DPU team](https://www.microsoft.com/en-us/research/project/program/).

## Python

Stored in the `python` subdirectory, published as the `dpu-utils` package.

### Installation

```bash
pip install dpu-utils
```
OR via the community-maintained Conda recipe:
```bash
conda install -c conda-forge dpu-utils
```

### Overview
Below you can find an overview of the utilities included. Detailed documentation
is provided at the docstring of each class.

##### Generic Utilities `dpu_utils.utils`
* [`ChunkWriter`](python/dpu_utils/utils/chunkwriter.py) provides a convenient API for writing output in multiple parts (chunks).
* [`RichPath`](python/dpu_utils/utils/richpath.py) an API that abstract local and Azure Blob paths in your code.
* [`*Iterator`](python/dpu_utils/utils/iterators.py) Wrappers that can parallelize and shuffle iterators.
* [`{load,save}_json[l]_gz`](python/dpu_utils/utils/dataloading.py) convenience API for loading and writing `.json[l].gz` files.
* [`git_tag_run`](python/dpu_utils/utils/gitlog.py) tags the current working directory git the state of the code.
* [`run_and_debug`](python/dpu_utils/utils/debughelper.py) when an exception happens, start a debug session. Usually a wrapper of `__main__`.

##### General Machine Learning Utilities `dpu_utils.mlutils`
* [`Vocabulary`](python/dpu_utils/mlutils/vocabulary.py) map elements into unique integer ids and back.
    Commonly used in machine learning models that work over discrete data (e.g. 
    words in NLP). Contains methods for converting an list of tokens into their
    "tensorized" for of integer ids.  
* [`BpeVocabulary`](python/dpu_utils/mlutils/bpevocabulary.py) a vocabulary for machine learning models that employs BPE (via `sentencepiece`).
* [`CharTensorizer`](python/dpu_utils/mlutils/chartensorizer.py) convert character sequences into into tensors, commonly used
    in machine learning models whose input is a list of characters.

##### Code-related Utilities `dpu_utils.codeutils`
* [`split_identifier_into_parts()`](python/dpu_utils/codeutils/identifiersplitting.py) split identifiers into subtokens on CamelCase and snake_case.
* [`Lattice`](python/dpu_utils/codeutils/lattice/lattice.py), [`CSharpLattice`](python/dpu_utils/codeutils/lattice/csharplattice.py) represent lattices and useful operations on lattices in Python.
* [`get_language_keywords()`](python/dpu_utils/codeutils/keywords/keywordlist.py) an API to retrieve the keyword tokens for many programming languages.
* [`language_candidates_from_suffix()`](python/dpu_utils/codeutils/filesuffix.py) a function to retrieve the candidate language given the file suffix.
* [`deduplication.DuplicateDetector`](python/dpu_utils/codeutils/deduplication/deduplication.py) API to detects (near)duplicates in codebases.
See also [here](#approximate-duplicate-code-detection) for a command line tool.
* [`treesitter.parser_for`](python/dpu_utils/codeutils/treesitter/parser.py) get [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) parser by language name.

##### TensorFlow 1.x Utilities `dpu_utils.tfutils`
* [`get_activation`](python/dpu_utils/tfutils/activation.py) retrieve activations function by name.
* [`GradRatioLoggingOptimizer`](python/dpu_utils/tfutils/gradratiologgingoptimizer.py) a wrapper around optimizers that logs the ratios of grad norms to parameter norms.
* [`TFVariableSaver`](python/dpu_utils/tfutils/tfvariablesaver.py) save TF variables in an object that can be pickled.

Unsorted segment operations following TensorFlow's [`unsorted_segment_sum`](https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum) operations:
* [`unsorted_segment_logsumexp`](python/dpu_utils/tfutils/unsortedsegmentops.py)
* [`unsorted_segment_log_softmax`](python/dpu_utils/tfutils/unsortedsegmentops.py)
* [`unsorted_segment_softmax`](python/dpu_utils/tfutils/unsortedsegmentops.py)

##### TensorFlow 2.x Utilities `dpu_utils.tf2utils`
* [`get_activation_function_by_name`](python/dpu_utils/tf2utils/activation.py) retrieve activation functions by name.
* [`gelu`](python/dpu_utils/tf2utils/activation.py) The GeLU activation function.
* [`MLP`](python/dpu_utils/tf2utils/mlp.py) An MLP layer.

Unsorted segment operations following TensorFlow's [`unsorted_segment_sum`](https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum) operations:
* [`unsorted_segment_logsumexp`](python/dpu_utils/tf2utils/unsorted_segment_ops.py)
* [`unsorted_segment_log_softmax`](python/dpu_utils/tf2utils/unsorted_segment_ops.py)
* [`unsorted_segment_softmax`](python/dpu_utils/tf2utils/unsorted_segment_ops.py)


##### TensorFlow Models `dpu_utils.tfmodels`
* [`SparseGGNN`](python/dpu_utils/tfmodels/sparsegnn.py) a sparse GGNN implementation.
* [`AsyncGGNN`](python/dpu_utils/tfmodels/asyncgnn.py) an asynchronous GGNN implementation.

These models have not been tested with TF 2.0.

##### PyTorch Utilities `dpu_utils.ptutils`
* [`BaseComponent`](python/dpu_utils/ptutils/basecomponent.py) a wrapper abstract class around `nn.Module` that 
   takes care of essential elements of most neural network components.
* [`ComponentTrainer`](python/dpu_utils/ptutils/basecomponent.py) a training loop for `BaseComponent`s.


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
