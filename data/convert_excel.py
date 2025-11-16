import pandas as pd
import os
import logging
import argparse # Import argparse

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

def convert_xlsx_to_csv(input_xlsx_file, output_csv_file, sheet_name=0, remove=False):
    """
    Converts a specific XLSX file to a cleaned CSV file.
    
    This function performs the following steps:
    1. Reads the Excel file.
    2. Sets the first row as the new header.
    3. Selects only the first 4 columns.
    4. Renames columns to 'pos' and 'type' for compatibility.
    5. Saves the result as a CSV.
    6. Optionally removes the original XLSX file.

    Args:
        input_xlsx_file (str): The path to the input XLSX file.
        output_csv_file (str): The path to the output CSV file.
        sheet_name (str or int, optional): The sheet to convert. Defaults to 0.
        remove (bool, optional): Whether to delete the original XLSX file. Defaults to False.
    """
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(input_xlsx_file, sheet_name=sheet_name)
        
        # --- Start of specific cleaning logic ---
        new_header = df.iloc[0] 
        df = df[1:] 
        df.columns = new_header 
        df = df.iloc[:, :4]
        df = df.rename(columns={df.columns[1]: "pos"})
        df = df.rename(columns={df.columns[2]: "type"})
        # --- End of specific cleaning logic ---

        df.to_csv(output_csv_file, index=False)
        logging.info(f"Successfully converted '{input_xlsx_file}' to '{output_csv_file}'.")
        
        if remove:
            os.remove(input_xlsx_file)
            logging.info(f"Removed original file: '{input_xlsx_file}'")
            
    except FileNotFoundError:
        logging.error(f"Error: Input file '{input_xlsx_file}' not found.")
    except Exception as e:
        logging.error(f"An error occurred while processing '{input_xlsx_file}': {e}")

# =============================================================================
# == MAIN EXECUTION
# =============================================================================
def main():
    """
    Main function to parse arguments and run the conversion script.
    """
    
    # --- Default list of sheets ---
    default_sheets = [
        "116", "117", "118", "119", "120", "121",
        "27952F", "27952G", "27952H", "27952J",
    ]

    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Convert and clean specific Excel files to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        default="BowTie/round-2/",
        help="Input directory containing the Excel (.xlsx) files. Default: 'BowTie/round-2/'"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for the CSV files. Defaults to the same as --input-dir if not specified."
    )
    
    parser.add_argument(
        "-s", "--sheets",
        nargs='+',
        default=default_sheets,
        help=f"List of sheet names (without .xlsx) to convert. Default: {default_sheets}"
    )
    
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Include this flag to delete the original .xlsx files after successful conversion."
    )
    
    args = parser.parse_args()

    # --- Set output directory logic ---
    input_dir = args.input_dir
    # If no output dir is specified, use the input dir
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"--- Starting Excel to CSV conversion ---")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    for sheet_name in args.sheets:
        input_path = os.path.join(input_dir, f"{sheet_name}.xlsx")
        output_path = os.path.join(output_dir, f"{sheet_name}.csv")
        
        convert_xlsx_to_csv(input_path, output_path, remove=args.remove) 

    logging.info("--- Conversion process finished. ---")


if __name__ == "__main__":
    main()