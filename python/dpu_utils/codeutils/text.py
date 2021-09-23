from itertools import count
import io
from typing import Tuple


def get_code_in_range(code_text: str, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> str:
    """
    Get the code text given a range in the form (start_line, start_column), (end_line, end_column).

    Notes:
        * This follow common IDE convention where lines and columns are 1-based (i.e., the first line
            is line 1 not line 0 as it would be in array indexing)
        * The range is inclusive (i.e., both start and end are included)

    :param code_text: the string representation of the code.
    :param start_pos: The starting position of the range in `code_text` as a tuple the form (start_line, start_column).
    :param end_pos: The end position of the range in `code_text` as a tuple the form (end_line, end_column).

    :return: the string representation of the target range.
    """
    (start_line, start_column), (end_line, end_column) = start_pos, end_pos
    assert start_line < end_line or (start_line == end_line and start_column <= end_column), "Invalid range"

    # Adjust `target_range` columns to be 0-based and the end_column to be exclusive
    start_column -= 1

    with io.StringIO(code_text) as input_sb, io.StringIO() as output_sb:
        for line_no in count(start=1):
            next_input_line = input_sb.readline()
            if len(next_input_line) == 0: # No bytes read
                raise ValueError("EOF reached before target_range.")

            if start_line <= line_no <= end_line:
                if start_line == end_line:
                    output_sb.write(next_input_line[start_column: end_column])
                    break
                elif line_no == start_line:
                    output_sb.write(next_input_line[start_column:])
                elif line_no == end_line:
                    output_sb.write(next_input_line[:end_column])
                    break
                else:
                    output_sb.write(next_input_line)
            elif line_no > end_line:
                raise Exception("Unreachable state.")
        return output_sb.getvalue()