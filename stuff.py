def create_concatenated_file(file_list, output_filename):
    """
    Concatenates the content of multiple files into a single file.

    This function reads each file from the provided list, adds a
    descriptive header, and appends its content to the specified
    output file. This is useful for creating a single context file
    for analysis or for providing context to language models.

    Args:
        file_list (list[str]): A list of relative file paths to concatenate.
        output_filename (str): The name of the file to save the combined content to.

    Sample Usage:
        files = ["src/main.py", "src/utils.py"]
        create_concatenated_file(files, "combined_code.txt")
    """
    try:
        with open(output_filename, "w", encoding="utf-8") as outfile:
            for file_path in file_list:
                # Create a clear header for each file's content
                header = f"--- FILE: {file_path} ---\n\n"
                outfile.write(header)

                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")  # Add spacing between files
                except FileNotFoundError:
                    outfile.write(f"*** File not found: {file_path} ***\n\n")
                except Exception as e:
                    outfile.write(f"*** Error reading file {file_path}: {e} ***\n\n")
        print(
            f"Successfully created '{output_filename}' with the content of {len(file_list)} files."
        )
    except IOError as e:
        print(
            f"Error: Could not write to the output file '{output_filename}'. Reason: {e}"
        )


if __name__ == "__main__":
    # The list of files you specified for debugging the Schelling simulation
    schelling_sim_files = [
        "simulations/schelling_sim/actions.py",
        "simulations/schelling_sim/components.py",
        "simulations/schelling_sim/environment.py",
        "simulations/schelling_sim/loader.py",
        "simulations/schelling_sim/systems.py",
        "simulations/schelling_sim/config/config.yml",
        "tests/simulations/schelling_sim/test_systems.py",
    ]

    output_file = "schelling_sim_context.txt"
    create_concatenated_file(schelling_sim_files, output_file)
