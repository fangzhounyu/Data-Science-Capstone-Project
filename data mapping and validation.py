import pandas as pd
import numpy as np
import os

def load_data(esg_data_path, mapping_path):
    """
    Load ESG data and company-Wind code mapping data.
    
    """
    try:
        esg_df = pd.read_csv(esg_data_path)
        print(f"Successfully loaded ESG data from {esg_data_path}")
    except FileNotFoundError:
        print(f"ESG data file not found: {esg_data_path}")
        return None, None
    
    try:
        mapping_df = pd.read_csv(mapping_path)
        print(f"Successfully loaded mapping data from {mapping_path}")
    except FileNotFoundError:
        print(f"Mapping data file not found: {mapping_path}")
        return esg_df, None
    
    return esg_df, mapping_df


def validate_mappings(esg_df, mapping_df):
    """
    Validate that all companies in ESG data have corresponding Wind codes.
    
    """
    if mapping_df is None:
        print("Mapping DataFrame is None. Cannot perform validation.")
        return None, esg_df[['Company']]
    
    merged_df = pd.merge(esg_df, mapping_df, on='Company', how='left')
    
    missing_mappings = merged_df[merged_df['Wind Code'].isnull()][['Company']]
    
    return merged_df, missing_mappings



def save_missing_mappings(missing_mappings, output_dir):
    """
    Save the list of companies missing Wind codes to a CSV file.
    
    """
    os.makedirs(output_dir, exist_ok=True)
    missing_path = os.path.join(output_dir, "missing_company_wind_codes.csv")
    missing_mappings.to_csv(missing_path, index=False)
    print(f"Saved missing mappings to {missing_path}")


def save_merged_data(merged_df, output_dir):
    """
    Save the merged ESG data with Wind codes to a CSV file.
    
    """
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, "esg_with_wind_codes.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"Saved merged ESG data with Wind codes to {merged_path}")



def main():
    # -------------------------------
    # Configuration
    # -------------------------------
    
    # Paths to data
    esg_data_path = "aggregated_esg_metrics.csv"  # Adjust path as needed
    mapping_path = "data/company_wind_code_mapping.csv"  # Adjust path as needed
    
    # Output directories
    validation_output_dir = "data_mapping_validation_output"
    merged_data_output_dir = "data_mapping_validation_output/merged_data"
    
    # -------------------------------
    # Load Data
    # -------------------------------
    
    print("Loading ESG data and mapping data...")
    esg_df, mapping_df = load_data(esg_data_path, mapping_path)
    if esg_df is None:
        print("Failed to load ESG data. Exiting.")
        return
    if mapping_df is None:
        print("Failed to load mapping data. Proceeding without mappings.")
    
    # -------------------------------
    # Validate Mappings
    # -------------------------------
    
    print("\nValidating company to Wind code mappings...")
    merged_df, missing_mappings = validate_mappings(esg_df, mapping_df)
    
    # -------------------------------
    # Handle Missing Mappings
    # -------------------------------
    
    if missing_mappings is not None and not missing_mappings.empty:
        print(f"\nFound {len(missing_mappings)} companies without Wind codes.")
        save_missing_mappings(missing_mappings, validation_output_dir)
        print("Please update the mapping file with the missing Wind codes and rerun the script.")
    else:
        print("\nAll companies have corresponding Wind codes.")
    
    # -------------------------------
    # Save Merged Data
    # -------------------------------
    
    if merged_df is not None:
        print("\nSaving merged ESG data with Wind codes...")
        save_merged_data(merged_df, merged_data_output_dir)
    
    print("\nData Mapping and Validation complete!")

if __name__ == "__main__":
    main()
