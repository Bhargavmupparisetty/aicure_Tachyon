import sys
import pandas as pd
import joblib

def predict_heart_rate(input_file, model_filename='gradient_boosting_model.joblib', output_file='results.csv'):
    # Load the input data
    input_data = pd.read_csv(input_file)

    # Load the trained model
    loaded_model = joblib.load(model_filename)

    # Extract features for prediction
    features = ['VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT',
                'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'SD1', 'SD2', 'sampen', 'higuci',
                'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD',
                'pNN25', 'pNN50', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR',
                'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR',
                'KURT_REL_RR', 'SKEW_REL_RR']

    
    x_data = data[features]
    x_uuid = data['uuid']
    # Make predictions on the test set using the loaded model
    y_pred_loaded = loaded_model.predict(x_data)
    df['HR'] = pd.DataFrame(y_pred_loaded)
    result_df = pd.concat([x_uuid, df['HR']], axis=1, ignore_index=True)
    result_df.dropna()
    # Set column names and index
    result_df.columns = ['uuid', 'HR']
    result_df.dropna()
    
    # Save the results to a CSV file
    input_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    predict_heart_rate(input_file)
