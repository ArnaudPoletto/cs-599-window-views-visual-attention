def find_and_get_next_char_index(string: str, substring: str) -> int:
    """
    Find the index of the substring in the string and return the index of the next character.

    Args:
        string (str): The string
        substring (str): The substring
    """
    find_index = string.find(substring)
    if find_index == -1:
        return -1
    
    next_char_index = find_index + len(substring)
    if next_char_index >= len(string):
        return -1

    return next_char_index
