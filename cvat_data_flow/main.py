from cvat_data_flow.src.cvat_data_flow import CVATDataFlow
from cvat_data_flow.src.utils.config_parser import Options
import os

def main():
    """
    Main function for the cvat_data_flow utility.
    """
    options = Options()

    cvat_data_flow = CVATDataFlow(
        url=options.url,
        login=options.login,
        password=options.password,
        raw_data_path=options.raw_data_path,
        projects_ids=options.projects_ids,
        tasks_ids=options.tasks_ids,
        dataset_format=options.format,
        split=options.split,
        labels_mapping=options.labels_mapping,
        debug=options.debug,
        labels_id_mapping=options.labels_id_mapping,
    )

    # download data
    if not options.only_build_dataset:
        if len(os.listdir(options.raw_data_path)) == 0:
            cvat_data_flow.download_data()
        else:
            cvat_data_flow.logger.info("Data already downloaded")

    # build dataset
    # check if the dataset is already built
    if not os.path.exists(f"{options.save_path}_{options.format}"):
        cvat_data_flow.build_dataset(save_path=options.save_path)
    else:
        cvat_data_flow.logger.info("Dataset already built")

if __name__ == '__main__':
    main()
