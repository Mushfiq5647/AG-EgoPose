# Define a dummy utils module with the necessary classes

import sys
import os

# Add the utils directory to the Python path
utils_path = '../utils'
sys.path.append(utils_path)


# Now, let's try to load the pickle file again with this dummy module
import pickle
import io

file_path = 'vocab.pkl'

try:
    with open(file_path, 'rb') as f:
        content = f.read()

    # Replace problematic module names and load content
    content_fixed = content.replace(b'copy_reg\r', b'copyreg').replace(b'copy_reg', b'copyreg').replace(b'\r', b'')
    fixed_file = io.BytesIO(content_fixed)
    vocab_data = pickle.load(fixed_file, encoding='latin1')

    # Save the file in a Python 3-compatible format
    output_file_path = 'vocab_py3.pkl'
    with open(output_file_path, 'wb') as f:
        pickle.dump(vocab_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    output_file_path

except Exception as e:
    print("Exception while pickling.",e)
    str(e)
