import pickle
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_reference_evaluation(input_pkl, output_pkl):
    """
    Process a reference evaluation .pkl file to apply truncation based on metadata.

    Args:
        input_pkl (str): Path to the input .pkl file.
        output_pkl (str): Path to save the modified .pkl file.
    """
    try:
        # Load the input .pkl file
        with open(input_pkl, "rb") as f:
            data = pickle.load(f)

        metadata = data.get("metadata", {})
        results = data.get("results", {})

        # Check for "dedicated_class" in metadata
        dedicated_class = metadata.get("dedicated_class", None)
        if dedicated_class:
            logger.info(f"Truncating results to only include the dedicated class: {dedicated_class}")
            truncated_results = {
                pair: metrics
                for pair, metrics in results.items()
                if dedicated_class in pair
            }
            data["results"] = truncated_results

        # Check for "reference_print_label_for_trauncation" in metadata
        print(metadata)
        reference_print_label = metadata.get("reference_print_label_for_trauncation", None)
        if reference_print_label:
            logger.info(f"Adding 'reference_print_label' to metadata: {reference_print_label}")
            metadata["reference_print_label"] = reference_print_label
        elif "reference_print_label" in metadata:
            pass
        else:
            # Use the original .pkl file name as the reference_print_label
            original_label = os.path.splitext(os.path.basename(input_pkl))[0]
            logger.info(f"No 'reference_print_label_for_trauncation' found. Using file name as 'reference_print_label': {original_label}")
            metadata["reference_print_label"] = original_label

        # Save the modified .pkl file
        with open(output_pkl, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Processed file saved to {output_pkl}")

    except Exception as e:
        logger.error(f"Error processing file {input_pkl}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a .pkl file with reference evaluation results.")
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to the input .pkl file.")
    parser.add_argument("--output_pkl", type=str, required=True, help="Path to save the modified .pkl file.")

    args = parser.parse_args()

    process_reference_evaluation(args.input_pkl, args.output_pkl)
