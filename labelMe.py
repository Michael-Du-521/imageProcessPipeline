import os
def create_sub_dir(parent_dir,sub_dir_name):
    # Define the path to the subdirectory
    sub_dir_path = os.path.join(parent_dir, sub_dir_name)
    # Create the subdirectory if it doesn't exist
    if not os.path.exists(sub_dir_path):
        os.mkdir(sub_dir_path)
    return sub_dir_path
