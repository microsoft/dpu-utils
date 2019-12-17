from collections import Counter, defaultdict
import typing
from typing import List, Dict, Set, TypeVar, Generic, Iterable, Tuple, Hashable, Optional, Callable
import re
import numpy as np

from dpu_utils.codeutils import get_language_keywords

from SetSimilaritySearch import all_pairs

DocumentId = TypeVar('DocumentId', bound=Hashable)


class DuplicateDetector(Generic[DocumentId]):
    """Detect near-duplicate code.

    This class accepts a list of tokens within the some code snippet. It then approximately finds all identifier
    tokens and creates a set T_1 with those tokens and a multiset T_2 with the same tokens.

    A file `i` is considered to be a duplicate with another one `j` if the Jaccard similarity of T_1^i and T_1^j
    is more than `set_similarity_threshold` and the Jaccard similarity of T_2^i and T_2^j is more than
    `multiset_similarity_threshold`. Documents with less than `min_num_tokens_per_document` are not considered.

    This follows the general principles in

        Sajnani H, Saini V, Svajlenko J, Roy CK, Lopes CV.
        SourcererCC: scaling code clone detection to big-code.
        In Software Engineering (ICSE), 2016
        IEEE/ACM 38th International Conference on 2016 May 14 (pp. 1157-1168)

    Sample usage:
        * Add all files (and their tokens) via `add_files()`
        * Call `compute_duplicates()`
        * If the goal is to retrieve a list of files to be excluded, instead use `compute_ids_to_exclude()`


    See also:
        Allamanis, Miltiadis. "The adverse effects of code duplication in machine learning models of code."
        Proceedings of the 2019 ACM SIGPLAN International Symposium on New Ideas, New Paradigms,
         and Reflections on Programming and Software. ACM, 2019.

    """

    IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

    def __init__(self, set_similarity_threshold: float=0.8, multiset_similarity_threshold: float=0.7,
                 min_num_tokens_per_document: int=20)-> None:
        self.__vocabulary = {}  # type: Dict[str, int]
        self.__set_similarity_threshold = set_similarity_threshold
        self.__multiset_similarity_threshold = multiset_similarity_threshold
        self.__min_num_tokens_per_document = min_num_tokens_per_document
        self.__document_keys = []  # type: List[DocumentId]
        self.__document_key_set = set()  # type: Set[DocumentId]
        self.__document_elements = []  # type: List[typing.Counter[int]]

    def __get_token_id(self, token: str) -> int:
        token_id = self.__vocabulary.get(token)
        if token_id is None:
            token_id = len(self.__vocabulary)
            self.__vocabulary[token] = token_id
        return token_id

    def add_file(self, id: DocumentId, tokens: List[str], language: Optional[str]=None) -> bool:
        """Add a file to be indexed by the duplicate detector."""
        id_tokens = Counter(self.__get_token_id(t) for t in tokens if self.IDENTIFIER_REGEX.match(t)
                            and (language is None or t not in get_language_keywords(language)))
        assert id not in self.__document_key_set
        if sum(id_tokens.values()) < self.__min_num_tokens_per_document:
            return False
        self.__document_keys.append(id)
        self.__document_key_set.add(id)
        self.__document_elements.append(id_tokens)
        return True

    def __duplicate_pairs(self)-> Iterable[Tuple[int, int]]:
        similar_pairs = all_pairs(self.__document_elements,
                                  similarity_func_name='jaccard',
                                  similarity_threshold=self.__set_similarity_threshold)
        for idx1, idx2, _ in similar_pairs:
            if self.__multiset_jaccard_similarity(idx1, idx2) >= self.__multiset_similarity_threshold:
                yield idx1, idx2

    def __multiset_jaccard_similarity(self, idx1: int, idx2: int)-> float:
        intersection_size = sum((self.__document_elements[idx1] & self.__document_elements[idx2]).values())
        union_size = sum((self.__document_elements[idx1] | self.__document_elements[idx2]).values())
        return float(intersection_size) / union_size

    def compute_duplicates(self) -> List[Set[DocumentId]]:
        """Compute the duplicates in the currently indexed documents.

        Make the incorrect but reasonable assumption that similarity is transitive.
        Compute the pairwise similar elements and add them into clusters."""

        clone_sets = []  # type: List[Set[DocumentId]]

        pairwise_relationships = defaultdict(list)  # type: Dict[int, List[int]]
        for idx1, idx2 in self.__duplicate_pairs():
            assert idx1 != idx2
            pairwise_relationships[idx1].append(idx2)
            pairwise_relationships[idx2].append(idx1)

        # Compute the transitive closure of this relationship
        documents_to_visit = set(pairwise_relationships.keys())  # type: Set[int]
        while len(documents_to_visit) > 0:
            current_idx = documents_to_visit.pop()

            current_idx_closure = {current_idx}
            visit_queue = list(pairwise_relationships[current_idx])
            while len(visit_queue) > 0:
                other_idx = visit_queue.pop()
                current_idx_closure.add(other_idx)
                documents_to_visit.discard(other_idx)

                # Add to queue
                visit_queue.extend(next_idx for next_idx in pairwise_relationships[other_idx]
                                   if next_idx in documents_to_visit)

            clone_sets.append(set(self.__document_keys[i] for i in current_idx_closure))
        return clone_sets

    def print_clone_set_stats(self, clone_sets: List[Set[DocumentId]]) -> None:
        total_num_files = len(self.__document_keys)
        num_cloned_files = sum(len(c) for c in clone_sets)
        print('Duplicated files: %.2f%%' % (num_cloned_files / total_num_files * 100.))
        print('Avg num of files per clones %.2f' % np.mean([len(c) for c in clone_sets]))
        print('Median num of files per clones %s' % np.median([len(c) for c in clone_sets]))

        print('Duplication Ratio %.2f%%' % ((num_cloned_files - len(clone_sets)) / total_num_files * 100))

    def compute_ids_to_exclude(self, keep_selector: Optional[Callable[[Set[DocumentId]], DocumentId]]=None) -> Set[DocumentId]:
        """Compute a set of document ids to discard in the currently indexed documents.

        :param keep_selector: a lambda that accepts a set of DocumentId's and returns the DocumentId to keep.
            If the DocumentId is not contained in input set, the whole cluster of duplicates is excluded.
            If keep_selector is None then it arbitrarily excludes one document id from each cluster of duplicates, and returns
            a set of the remaining document ids to exclude in order to de-duplicate your data.
        """
        duplicate_clusters = self.compute_duplicates()
        # remove one document from each duplicate set to keep
        for cluster in duplicate_clusters:
            if keep_selector is None:
                cluster.pop()   # Remove arbitrary element
            else:
                document_to_keep = keep_selector(cluster)
                cluster.discard(document_to_keep)

        # flatten out the lists of sets into one superset, each element being a document_id that you will discard
        return set.union(*duplicate_clusters)
