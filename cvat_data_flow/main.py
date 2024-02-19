from src.cvat_data_flow import CVATDataFlow
from src.config_parser import Options

def main():
    """
    Main function for the cvat_data_flow utility.
    """
    options = Options()

    cvat_data_flow = CVATDataFlow(
        url=options.url,
        login=options.login,
        password=options.password,
        save_path=options.save_path,
        projects_ids=options.projects_ids,
        tasks_ids=options.tasks_ids,
        only_build_dataset=options.only_build_dataset,
        dataset_format=options.format,
        split=options.split,
        labels_mapping=options.labels_mapping,
        debug=options.debug
    )
    cvat_data_flow.download_data()
    cvat_data_flow.build_dataset()

if __name__ == '__main__':
    main()
