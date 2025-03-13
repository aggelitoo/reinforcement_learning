unique_dict = {}
for state, target in data:
    # Convert the matrix to a hashable representation.
    # Option 1: Using tobytes() (ensures shape and dtype are the same)
    key = (state.tobytes(), target)
    
    # Option 2: Alternatively, you could do:
    # key = (tuple(matrix.flatten()), integer)
    
    if key not in unique_dict:
        unique_dict[key] = (state, target)

unique_data = list(unique_dict.values())