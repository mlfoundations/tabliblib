from collections import defaultdict


def path_to_str(path: str) -> str:
    """Convert a path to a string without slashes."""
    return path.replace("/", "__").replace(".parquet", "")


def find_covering_set(file_string_tuples):
    file_to_strings = defaultdict(set)
    string_to_files = defaultdict(set)
    for file, string in file_string_tuples:
        file_to_strings[file].add(string)
        string_to_files[string].add(file)

    all_strings = set(string for _, string in file_string_tuples)
    selected_files = set()
    output_tuples = []

    while all_strings:
        best_file = None
        strings_covered = set()
        # Identify the file that covers the most uncovered strings
        for file, strings in file_to_strings.items():
            covered = strings & all_strings
            if len(covered) > len(strings_covered):
                best_file = file
                strings_covered = covered

        # Update the sets and output if a file was found
        if best_file:
            selected_files.add(best_file)
            all_strings -= strings_covered
            for s in strings_covered:
                output_tuples.extend([(best_file, s)])
            # Remove the selected file to prevent it from being chosen again
            del file_to_strings[best_file]
        else:
            # No more files can contribute to covering strings, break out of the loop
            break

    # Return only the tuples corresponding to selected files
    return output_tuples
