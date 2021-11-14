# Put the code for your API here.
"""
ML Pipeline
"""
import argparse
import src.basic_cleaning
import src.train_model
import src.check_slices 
import logging


def execute(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "basic_cleaning":
        logging.info("Basic cleaning procedure started")
        src.basic_cleaning.execute_cleaning()

    if args.action == "all" or args.action == "train_model":
        logging.info("Train/Test model procedure started")
        src.train_model.train_test_model()

    if args.action == "all" or args.action == "check_slice":
        logging.info("Slice score check procedure started")
        src.check_slices.check_score()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML Training Pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["basic_cleaning",
                 "train_model",
                 "check_slice",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    execute(main_args)