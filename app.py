import streamlit as st
import pandas as pd
import joblib
import os

# Page settings
st.set_page_config(
    page_title="Employee Performance Prediction",
    layout="wide"
)

# Load dataset and trained model
base_df = pd.read_csv("employee_training_data.csv")
model = joblib.load("employee_performance_model.pkl")

# Add names to sample employees
base_df["Name"] = base_df["Employee_ID"].apply(
    lambda x: f"Sample Employee {x}"
)

columns = [
    "Employee_ID",
    "Name",
    "Attendance",
    "Productivity",
    "Task_Completion",
    "Average_Score",
    "Performance_Category"
]

base_df = base_df[columns]

# Load previously saved employees
if os.path.exists("employee_records.csv"):
    saved_df = pd.read_csv("employee_records.csv")
else:
    saved_df = pd.DataFrame(columns=columns)

all_df = pd.concat(
    [base_df, saved_df],
    ignore_index=True
)

st.title("AI-Based Employee Performance Prediction System")
st.write("Welcome to the Employee Performance Prediction System")

# Add new employee
st.header("Add New Employee")

with st.form("employee_form"):
    name = st.text_input("Enter Employee Name")

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    productivity = st.number_input(
        "Productivity (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    task_completion = st.number_input(
        "Task Completion (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    save_button = st.form_submit_button("Save Employee")

if save_button:
    if name.strip() == "":
        st.warning("Please enter employee name.")
    else:
        new_employee = pd.DataFrame({
            "Attendance": [attendance],
            "Productivity": [productivity],
            "Task_Completion": [task_completion]
        })

        prediction = model.predict(new_employee)[0]

        average_score = (
            attendance + productivity + task_completion
        ) / 3

        new_id = int(all_df["Employee_ID"].max()) + 1

        new_record = pd.DataFrame([{
            "Employee_ID": new_id,
            "Name": name.strip(),
            "Attendance": attendance,
            "Productivity": productivity,
            "Task_Completion": task_completion,
            "Average_Score": average_score,
            "Performance_Category": prediction
        }])

        saved_df = pd.concat(
            [saved_df, new_record],
            ignore_index=True
        )

        saved_df.to_csv("employee_records.csv", index=False)

        st.success(
            f"Employee saved successfully! Employee ID: {new_id}"
        )

        st.rerun()

# Display employee records
st.divider()
st.header("Employee Records")

all_df = pd.concat(
    [base_df, saved_df],
    ignore_index=True
)

st.dataframe(all_df, use_container_width=True)

st.write("Total Employees:", len(all_df))

# View employee performance
st.divider()
st.header("View Employee Performance")

employee_id = st.selectbox(
    "Select Employee ID",
    all_df["Employee_ID"].tolist(),
    format_func=lambda x: (
        f"{x} - {all_df.loc[all_df['Employee_ID'] == x, 'Name'].iloc[0]}"
    )
)

employee = all_df[
    all_df["Employee_ID"] == employee_id
].iloc[0]

st.subheader("Employee Details")

st.write("Employee Name:", employee["Name"])
st.write("Attendance:", employee["Attendance"], "%")
st.write("Productivity:", employee["Productivity"], "%")
st.write("Task Completion:", employee["Task_Completion"], "%")
st.write("Average Score:", round(employee["Average_Score"], 2))

st.subheader("Performance Result")

st.success(
    "Performance Category: " +
    str(employee["Performance_Category"])
)

st.caption(
    "Predictions are based on the sample training dataset "
    "and are intended for academic demonstration."
)
import os
import joblib

# Find the folder containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load employee model
model_path = os.path.join(BASE_DIR, "employee_model.pkl")

model = joblib.load(model_path)

st.success("Employee model loaded successfully!")

st.header("Employee Performance Prediction")

# Employee details
name = st.text_input("Enter Employee Name")

attendance = st.number_input(
    "Attendance", min_value=0, max_value=100, value=80
)

productivity = st.number_input(
    "Productivity", min_value=0, max_value=100, value=80
)

task_completion = st.number_input(
    "Task Completion", min_value=0, max_value=100, value=80
)

# Prediction button
if st.button("Predict Performance"):

    if name.strip() == "":
        st.warning("Please enter employee name.")

    else:
        input_data = pd.DataFrame(
            [[attendance, productivity, task_completion]],
            columns=["Attendance", "Productivity", "Task_Completion"]
        )

        prediction = model.predict(input_data)[0]

        st.success("Employee: " + name)
        st.write("Predicted Performance:", prediction)

        # Save employee information
        employee_record = pd.DataFrame([{
            "Name": name,
            "Attendance": attendance,
            "Productivity": productivity,
            "Task_Completion": task_completion,
            "Predicted_Performance": prediction
        }])

        file_name = os.path.join(BASE_DIR, "employee_records.csv")

        if os.path.exists(file_name):
            employee_record.to_csv(
                file_name, mode="a", header=False, index=False
            )
        else:
            employee_record.to_csv(file_name, index=False)

        st.success("Employee record saved successfully!")

# Display all employee records
# All Employee Records
st.subheader("All Employee Records")

file_name = os.path.join(BASE_DIR, "employee_records.csv")

if os.path.exists(file_name):

    records = pd.read_csv(file_name)

    # EDIT EMPLOYEE
    st.subheader("Edit Employee")

    if not records.empty:
        edit_id = st.selectbox(
            "Select Employee ID to Edit",
            records.index,
            format_func=lambda i: str(records.loc[i, "Employee_ID"]),
            key="edit_employee"
        )

        edited_name = st.text_input(
            "Employee Name",
            value=str(records.loc[edit_id, "Name"])
        )

        edited_attendance = st.number_input(
            "Attendance",
            min_value=0,
            max_value=100,
            value=int(records.loc[edit_id, "Attendance"])
        )

        edited_productivity = st.number_input(
            "Productivity",
            min_value=0,
            max_value=100,
            value=int(records.loc[edit_id, "Productivity"])
        )

        edited_task = st.number_input(
            "Task Completion",
            min_value=0,
            max_value=100,
            value=int(records.loc[edit_id, "Task_Completion"])
        )

        if st.button("Update Employee"):
            records.loc[edit_id, "Name"] = edited_name
            records.loc[edit_id, "Attendance"] = edited_attendance
            records.loc[edit_id, "Productivity"] = edited_productivity
            records.loc[edit_id, "Task_Completion"] = edited_task

            records.loc[edit_id, "Average_Score"] = (
                edited_attendance +
                edited_productivity +
                edited_task
            ) / 3

            score = records.loc[edit_id, "Average_Score"]

            if score >= 80:
                category = "High"
            elif score >= 60:
                category = "Average"
            else:
                category = "Low"

            records.loc[edit_id, "Performance_Category"] = category

            records.to_csv(file_name, index=False)
            st.success("Employee updated successfully!")
            st.rerun()

    # DELETE EMPLOYEE
    st.subheader("Delete Employee")

    if not records.empty:
        delete_id = st.selectbox(
            "Select Employee ID to Delete",
            records.index,
            format_func=lambda i: str(records.loc[i, "Employee_ID"]),
            key="delete_employee"
        )

        if st.button("Delete Employee"):
            records = records.drop(delete_id)
            records.to_csv(file_name, index=False)
            st.success("Employee deleted successfully!")
            st.rerun()

    # DISPLAY ALL RECORDS
    st.subheader("Updated Employee Records")
    st.dataframe(records, use_container_width=True)

else:
    st.info("No employee records available yet.")