"""
R.E.T.A.I.N. Interactive Sample Prediction & Verification Script
Run this script to test employee flight risk predictions without needing a frontend.
"""

from src.api_gateway import predict_flight_risk, load_model_bundle

def main():
    print("=" * 65)
    print("      R.E.T.A.I.N. ML Engine — Flight Risk Verification")
    print("=" * 65)

    # Load model bundle
    load_model_bundle()
    print("\n[+] Model loaded successfully from 'models/retain_rf_model.pkl'.\n")

    # Sample Employee Profiles
    profiles = [
        {
            "name": "Employee A (High Stressor Profile)",
            "data": {
                "Age": 29,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 350,
                "Department": "Sales",
                "DistanceFromHome": 28,
                "Education": 2,
                "EducationField": "Marketing",
                "EnvironmentSatisfaction": 1,
                "Gender": "Male",
                "HourlyRate": 40,
                "JobInvolvement": 1,
                "JobLevel": 1,
                "JobRole": "Sales Executive",
                "JobSatisfaction": 1,
                "MaritalStatus": "Single",
                "MonthlyIncome": 2700,
                "MonthlyRate": 14000,
                "NumCompaniesWorked": 6,
                "OverTime": "Yes",
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 1,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 5,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 1,
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 0
            }
        },
        {
            "name": "Employee B (Moderate Profile)",
            "data": {
                "Age": 36,
                "BusinessTravel": "Travel_Rarely",
                "DailyRate": 800,
                "Department": "Research & Development",
                "DistanceFromHome": 8,
                "Education": 3,
                "EducationField": "Medical",
                "EnvironmentSatisfaction": 3,
                "Gender": "Female",
                "HourlyRate": 65,
                "JobInvolvement": 3,
                "JobLevel": 2,
                "JobRole": "Research Scientist",
                "JobSatisfaction": 3,
                "MaritalStatus": "Married",
                "MonthlyIncome": 5200,
                "MonthlyRate": 16000,
                "NumCompaniesWorked": 2,
                "OverTime": "No",
                "PercentSalaryHike": 14,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StockOptionLevel": 1,
                "TotalWorkingYears": 10,
                "TrainingTimesLastYear": 3,
                "WorkLifeBalance": 3,
                "YearsAtCompany": 6,
                "YearsInCurrentRole": 4,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 4
            }
        },
        {
            "name": "Employee C (High Retention Profile)",
            "data": {
                "Age": 45,
                "BusinessTravel": "Non-Travel",
                "DailyRate": 1200,
                "Department": "Research & Development",
                "DistanceFromHome": 10,
                "Education": 6,
                "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": 1,
                "Gender": "Female",
                "HourlyRate": 120,
                "JobInvolvement": 4,
                "JobLevel": 4,
                "JobRole": "Manager",
                "JobSatisfaction": 4,
                "MaritalStatus": "Married",
                "MonthlyIncome": 500,
                "MonthlyRate": 100,
                "NumCompaniesWorked": 10,
                "OverTime": "No",
                "PercentSalaryHike": 18,
                "PerformanceRating": 4,
                "RelationshipSatisfaction": 4,
                "StockOptionLevel": 2,
                "TotalWorkingYears": 1,
                "TrainingTimesLastYear": 4,
                "WorkLifeBalance": 0,
                "YearsAtCompany": 1,
                "YearsInCurrentRole": 10,
                "YearsSinceLastPromotion": 2,
                "YearsWithCurrManager": 9
            }
        }
    ]

    for item in profiles:
        print("-" * 65)
        print(f" PROFILE: {item['name']}")
        print("-" * 65)
        
        result = predict_flight_risk(item["data"])
        
        prob_pct = result.flight_risk_probability * 100
        print(f"  • Flight Risk Probability : {prob_pct:.2f}%")
        print(f"  • Risk Tier Evaluation   : [{result.risk_level}]")
        print(f"  • Key Stressors Identified: {', '.join(result.top_stressors)}")
        print()

    print("=" * 65)
    print(" Verification complete. All predictions generated successfully.")
    print("=" * 65)

if __name__ == "__main__":
    main()
