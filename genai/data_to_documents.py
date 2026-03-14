import pandas as pd
from sqlalchemy import create_engine
from langchain_core.documents import Document

def extract_and_convert_data():
    engine = create_engine('postgresql://postgres:Preet%403753@localhost:5432/insurance_db')
    
    # Read the data, limiting to a sample size to keep local vector DB manageable for hackathon
    print("Extracting data from PostgreSQL...")
    limit = 1000
    df_features = pd.read_sql(f'SELECT * FROM processed_feature_data LIMIT {limit}', engine)
    df_clusters = pd.read_sql(f'SELECT risk_segment FROM driver_risk_clusters ORDER BY original_row_index LIMIT {limit}', engine)
    df_claims = pd.read_sql(f'SELECT claim_probability FROM claim_cost_predictions ORDER BY original_row_index LIMIT {limit}', engine)
    
    # Join logically by row order
    df = df_features.join(df_clusters).join(df_claims)
    
    docs = []
    print("Converting to LangChain documents...")
    for idx, row in df.iterrows():
        # Example document format as requested
        text = (f"Driver aged {row['driver_age']} - Gender: {row['gender']} "
                f"- Driving experience: {row['driving_experience']} "
                f"- Credit Score: {row['credit_score']:.2f} "
                f"- Driver risk score: {row['driver_risk_score']} "
                f"- Vehicle age category: {row['vehicle_age']} years "
                f"- Annual mileage: {row['annual_mileage']} km "
                f"- Predicted claim probability: {row['claim_probability']:.2f} "
                f"- Risk segment: {row['risk_segment']} risk cluster.")
        
        metadata = {
            "age_group": str(row['driver_age']),
            "risk_cluster": str(row['risk_segment']),
            "vehicle_age": str(row['vehicle_age'])
        }
        docs.append(Document(page_content=text, metadata=metadata))
        
    print(f"Created {len(docs)} document objects.")
    return docs

if __name__ == "__main__":
    docs = extract_and_convert_data()
    print("Ready to embed!")
