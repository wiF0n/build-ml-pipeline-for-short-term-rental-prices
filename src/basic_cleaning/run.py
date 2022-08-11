"""
Download the raw dataset from W&B,
apply some basic data cleaning and
export the result to a new artifact
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(args):
    """
    Main function to execute
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    input_artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Load the input artifact to pandas df
    input_df = pd.read_csv(input_artifact_local_path)

    # Drop data that are not within defined geolocation
    geo_idx = input_df['longitude'].between(
        -74.25, -73.50) & input_df['latitude'].between(40.5, 41.2)
    geo_df = input_df[geo_idx].copy()

    # Drop outliers for price
    prep_price_idx = geo_df["price"].between(args.min_price, args.max_price)
    prep_price_df = geo_df[prep_price_idx].copy()

    # Drop outliers for minimum_nights
    prep_minimum_nights_idx = prep_price_df["minimum_nights"].between(
        args.min_minimum_nights, args.max_minimum_nights)
    prep_minimum_nights_df = prep_price_df[prep_minimum_nights_idx].copy()

    # Save final dataset as csv
    final_df_name = "clean_sample.csv"
    prep_minimum_nights_df.to_csv(final_df_name, index=False)

    # Create output artifact
    output_artifact = wandb.Artifact(args.output_artifact,
                                     type=args.output_type,
                                     description=args.output_description)

    # Add file to the output artifact and log it to W&B
    output_artifact.add_file(final_df_name)
    run.log_artifact(output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument("--input_artifact",
                        type=str,
                        help="Input dataset name from W&B.",
                        required=True)

    parser.add_argument("--output_artifact",
                        type=str,
                        help="Output (cleaned) dataset name.",
                        required=True)

    parser.add_argument("--output_type",
                        type=str,
                        help="Type of the output artifact (dataset)",
                        required=True)

    parser.add_argument("--output_description",
                        type=str,
                        help="Description of the output artifact (dataset)",
                        required=True)

    parser.add_argument("--min_price",
                        type=float,
                        help="Minimum price for one night (in $)",
                        required=True)

    parser.add_argument("--max_price",
                        type=float,
                        help="Maximum price for one night (in $)",
                        required=True)

    parser.add_argument("--min_minimum_nights",
                        type=int,
                        help="Minimum of minimum days of stay",
                        required=True)

    parser.add_argument("--max_minimum_nights",
                        type=int,
                        help="Maximum of minimum days of stay",
                        required=True)

    my_args = parser.parse_args()

    main(my_args)
