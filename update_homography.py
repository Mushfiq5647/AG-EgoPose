import os


def load_homography_from_file(file_path):
    """Read a homography vector from a file and return it as a list of floats."""
    with open(file_path, 'r') as f:
        homography = list(map(float, f.read().strip().split()))
    assert len(homography) == 9, f"Expected 9 elements, got {len(homography)} in {file_path}"
    return homography


def create_homography_sequence_and_save(num_frames, input_folder, output_folder, sequence_length=15,
                                        identity_matrix=[1, 0, 0, 0, 1, 0, 0, 0, 1]):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for t in range(num_frames):
        # Initialize the homography vector for the current frame
        h_t = []

        # Add homographies from the previous 14 frames or identity if not available
        for offset in range(sequence_length - 1, 0, -1):
            if t - offset >= 0:
                # Load the previous frame's homography
                file_path = os.path.join(input_folder, f"h{t - offset}.txt")
                h_t.extend(load_homography_from_file(file_path))
            else:
                # Use identity matrix if there is no previous frame
                h_t.extend(identity_matrix)

        # Append current frame's homography
        current_file_path = os.path.join(input_folder, f"h{t}.txt")
        h_t.extend(load_homography_from_file(current_file_path))

        # Ensure the vector length is 135
        assert len(h_t) == 135, f"Expected 135 values, got {len(h_t)} for frame {t}"

        # Save the 135-length homography vector to a file
        output_file_path = os.path.join(output_folder, f"h{t}.txt")
        with open(output_file_path, 'w') as f:
            f.write(" ".join(map(str, h_t)))
        print(f"Saved {output_file_path}")

def rename_extended_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith("_extended.txt"):
            # Extract the base name (e.g., `h0` from `h0_extended.txt`)
            new_name = filename.replace("_extended.txt", ".txt")
            # Define full file paths
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")


# # Example usage
# directory = "you2me_ds_release_cmu (1)/cmu/9-convo6/features/homography"  # Replace with your directory path
# rename_extended_files(directory)

input_folder = "D:/Dataset/EgoPW_dataset/EgoPW_dataset_release/ayush_new/kitchen1/homography"  # Directory containing h0.txt, h1.txt, ..., hn.txt
output_folder = "D:/Dataset/EgoPW_dataset/EgoPW_dataset_release//ayush_new/kitchen1/homography_updated"
homography_extensions = '.txt'

# Get all files in the directory with specified extensions and count them
num_frames = len([file for file in os.listdir(input_folder) if file.lower().endswith(homography_extensions)])

print(f"Number of homography files in the directory: {num_frames}")

# Generate and save the extended homographies
create_homography_sequence_and_save(num_frames, input_folder, output_folder)

