from typing import Tuple


class CoordinatesBuffer:
    """
    A buffer to store the last coordinates.
    """

    def __init__(self, max_length: int) -> None:
        """
        Initialize the buffer.

        Args:
            max_length (int): The maximum length of the buffer.
        """
        self.max_length = max_length
        self.buffer = []

    def add(self, coordinates: Tuple[float, float]) -> None:
        """
        Add coordinates to the buffer.

        Args:
            coordinates (Tuple[float, float]): The coordinates to add.
        """
        if len(self.buffer) == self.max_length:
            self.buffer.pop(0)
        self.buffer.append(coordinates)

    def get_most_recent(self) -> Tuple[float, float]:
        """
        Get the most recent coordinates.

        Returns:
            Tuple[float, float]: The most recent coordinates.
        """
        return self.buffer[-1]

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
            bool: True if the buffer is empty, False otherwise.
        """
        return len(self.buffer) == 0

    def __iter__(self):
        """
        Iterate over the buffer.

        Returns:
            Iterator: An iterator over the buffer.
        """
        return iter(self.buffer)

    def __getitem__(self, index: int) -> Tuple[float, float]:
        """
        Get the coordinates at the given index.

        Args:
            index (int): The index of the coordinates to get.

        Returns:
            Tuple[float, float]: The coordinates at the given index.
        """
        return self.buffer[index]
