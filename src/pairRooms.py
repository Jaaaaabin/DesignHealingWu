from const_project import DIRS_DATA_TOPO

def read_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def get_space_pairs_via_connections(spaces_file, connections_file, output_file, pair_connection_type=''):
    
    # Read the space and connection data from files
    spaces = read_file(spaces_file)
    connection_list = read_file(connections_file)
    connections_per_space = [c.replace(',', ' ').split() for c in connection_list]

    # check and remove repeated values in the connection list.
    connections_per_space = [list(set(c)) for c in connections_per_space]

    # Create a dictionary to associate doors with spaces
    dict_connection_to_spaces = {}
    for space, connections in zip(spaces, connections_per_space):
        for connection in connections:
            if connection in dict_connection_to_spaces:
                dict_connection_to_spaces[connection].append(space)
            else:
                dict_connection_to_spaces[connection] = [space]

    # Create a set to store unique pairs of spaces
    space_pairs = set()

    # Iterate through the door to space mapping
    for spaces in dict_connection_to_spaces.values():
        if len(spaces) == 2:  # Assuming one door connects exactly two spaces
            space_pair = tuple(sorted(spaces))
            space_pairs.add(space_pair)
        else:
            print(f"The {pair_connection_type} connects more than two spaces: {spaces}")

    # Write the pairs to the output file
    with open(output_file, 'w') as out_file:
        for pair in space_pairs:
            out_file.write(f"{pair[0]}, {pair[1]}\n")

    return space_pairs

# Specify the input and output file names
filepath = DIRS_DATA_TOPO

spaces_file = filepath + r'\collected_topology_space_host.txt'

doors_file = filepath + r'\collected_topology_space_doors.txt'
separationlines_file = filepath + r'\collected_topology_space_separationlines.txt'

output_file_by_door = filepath + r'\space_pairs_by_doors.txt'
output_file_by_separationline = filepath + r'\space_pairs_by_separationlines.txt'

def getRoomPairs():
    
    # Use the function to get space pairs and write them to a file
    get_space_pairs_via_connections(
        spaces_file,
        doors_file,
        output_file_by_door,
        pair_connection_type='Door')

    get_space_pairs_via_connections(
        spaces_file,
        separationlines_file,
        output_file_by_separationline,
        pair_connection_type='SeparationLine')