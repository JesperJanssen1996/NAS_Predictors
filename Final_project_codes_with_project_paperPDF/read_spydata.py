import pickle
import tarfile

def load_spydata(file_path):
    """
    Load a .spydata file and return its variables as a dictionary.
    Args:
        file_path (str): Path to the .spydata file.
    Returns:
        dict: A dictionary containing variable names as keys and their corresponding data as values.
    """
    variables = {}
    with tarfile.open(file_path, mode='r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.pickle'):
                # Extract and load each pickle file
                f = tar.extractfile(member)
                variable_name = member.name.replace('.pickle', '')
                variables[variable_name] = pickle.load(f)
                print(f"Loaded variable: {variable_name}")
    return variables
