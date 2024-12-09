import argparse
import os

# Share URL: "https://drive.google.com/file/d/1j8HjTdK48t_BgI9q6HJubSdomPFsaqAL/view?usp=sharing"
DEFAULT_FILE_ID = "1j8HjTdK48t_BgI9q6HJubSdomPFsaqAL"
DEFAULT_INTERMEDIATE_PATH = "data/raw/polygon.zip"
DEFAULT_OUTPUT_PATH = "data/raw/"


def get_args():
    """
    Get command line arguments.

    Returns:
    - args (argparse.Namespace): Command line arguments
    """
    parser = argparse.ArgumentParser(description="Download Polygon dataset")
    parser.add_argument(
        "--file_id",
        type=str,
        default=DEFAULT_FILE_ID,
        help="ID of the file",
    )
    parser.add_argument(
        "--intermediate_path",
        type=str,
        default=DEFAULT_INTERMEDIATE_PATH,
        help="Path to save the zip file before unzipping",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if the file exists",
    )
    return parser.parse_args()


def ensure_folder_exists(folder_path):
    """
    Ensure that the folder exists. If not, create it.

    Args:
    - folder_path (str): Path to the folder

    Returns:
    - None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def main(config):
    # make sure the output folder exists
    ensure_folder_exists(os.path.dirname(config["output_path"]))

    if os.path.exists(config["intermediate_path"]) and not config["force"]:
        print(
            f"Intermediate file already exists at {config['intermediate_path']}. Skipping download."
        )
        print("Set --force to redownload and overwrite the file.")
        return

    if len(os.listdir(config["output_path"])) > 0 and not config["force"]:
        print(f"{config['output_path']} is nonempty. Skipping download.")
        print("Set --force to redownload and overwrite the contents.")
        return

    print("Downloading Polygon dataset...")
    os.system(f"gdown {config['file_id']} --output {config['intermediate_path']}")

    print("Unzipping the file...")
    # Security risk, but necessary for large files
    os.environ["UNZIP_DISABLE_ZIPBOMB_DETECTION"] = "TRUE"
    os.system(f"unzip {config['intermediate_path']} -d {config['output_path']}")


if __name__ == "__main__":
    args = get_args()

    config = vars(args)
    main(config)
