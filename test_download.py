from utils.download import *
from utils.file_types import File_type

def main():
    DownloardAndExtractFile().download_and_extract_file(data_url.POWER_PLANT_DATASET_URL, File_type.ZIP)

if __name__ == '__main__':
    main()
