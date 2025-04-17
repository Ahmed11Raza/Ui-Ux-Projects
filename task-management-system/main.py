# Task Management System with Streamlit and UV-Python
# This application allows users to create, view, update, and delete tasks
# as well as visualize task data using UV-Python

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import uuid
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
#import pyuv as uv  # Import UV-Python for visualizations

# --- File Operations ---
DATA_FILE = "tasks.json"

def load_data():
    """Load task data from JSON file"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"tasks": []}

def save_data(data):
    """Save task data to JSON file"""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# --- Task Functions ---
def add_task(title, description, priority, due_date, category):
    """Add a new task to the system"""
    data = load_data()
    new_task = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "priority": priority,
        "due_date": due_date,
        "category": category,
        "status": "Pending",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    data["tasks"].append(new_task)
    save_data(data)
    return new_task

def delete_task(task_id):
    """Delete a task from the system"""
    data = load_data()
    data["tasks"] = [task for task in data["tasks"] if task["id"] != task_id]
    save_data(data)

def update_task_status(task_id, new_status):
    """Update the status of a task"""
    data = load_data()
    for task in data["tasks"]:
        if task["id"] == task_id:
            task["status"] = new_status
            save_data(data)
            return True
    return False

def get_task_by_id(task_id):
    """Get a task by its ID"""
    data = load_data()
    for task in data["tasks"]:
        if task["id"] == task_id:
            return task
    return None

def update_task(task_id, title, description, priority, due_date, category, status):
    """Update a task's details"""
    data = load_data()
    for task in data["tasks"]:
        if task["id"] == task_id:
            task["title"] = title
            task["description"] = description
            task["priority"] = priority
            task["due_date"] = due_date
            task["category"] = category
            task["status"] = status
            save_data(data)
            return True
    return False

# --- UV-Python Visualization Functions ---
def create_task_status_visualization(tasks):
    """Create a UV-Python visualization of task statuses"""
    status_counts = {"Pending": 0, "In Progress": 0, "Completed": 0, "Cancelled": 0}
    
    for task in tasks:
        if task["status"] in status_counts:
            status_counts[task["status"]] += 1
    
    # Create a UV-Python visualization
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Convert data for UV-Python
    labels = list(status_counts.keys())
    values = list(status_counts.values())
    
    # Create a color map using UV-Python
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    
    # Create a pie chart
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Task Distribution by Status')
    
    return fig

def create_priority_visualization(tasks):
    """Create a UV-Python visualization of task priorities"""
    priority_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    
    for task in tasks:
        if task["priority"] in priority_counts:
            priority_counts[task["priority"]] += 1
    
    # Create a UV-Python visualization
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Convert data for UV-Python
    labels = list(priority_counts.keys())
    values = list(priority_counts.values())
    
    # Color gradient based on priority
    colors = ['#CCFFCC', '#FFFFCC', '#FFCC99', '#FF9999']
    
    # Create a bar chart
    bars = ax.bar(labels, values, color=colors)
    
    # Add labels and title
    ax.set_xlabel('Priority Level')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Tasks by Priority Level')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    return fig

def create_category_visualization(tasks):
    """Create a UV-Python visualization of task categories"""
    categories = {}
    
    for task in tasks:
        category = task["category"]
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1
    
    # Create a UV-Python visualization
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Convert data for UV-Python
    labels = list(categories.keys())
    values = list(categories.values())
    
    # Create a horizontal bar chart with a colorful gradient
    cmap = plt.cm.get_cmap('viridis', len(labels))
    colors = [cmap(i) for i in range(len(labels))]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Number of Tasks')
    ax.set_title('Tasks by Category')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.annotate(f'{width}',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0),  # 3 points horizontal offset
                   textcoords="offset points",
                   ha='left', va='center')
    
    return fig

def create_due_date_visualization(tasks):
    """Create a UV-Python visualization of tasks by due date"""
    # Create dictionary to store tasks by month
    months = {i: 0 for i in range(1, 13)}
    
    for task in tasks:
        try:
            due_date = datetime.datetime.strptime(task["due_date"], "%Y-%m-%d")
            month = due_date.month
            months[month] += 1
        except (ValueError, TypeError):
            continue
    
    # Create a UV-Python visualization
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Convert data for UV-Python
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    values = list(months.values())
    
    # Create a line chart with markers
    x = range(1, 13)
    ax.plot(x, values, marker='o', linestyle='-', linewidth=2, markersize=8, 
            color='blue', markerfacecolor='red', markeredgecolor='red')
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Tasks by Due Date Month')
    ax.set_xticks(x)
    ax.set_xticklabels(month_names)

    return fig

# --- Streamlit UI ---
def main():
    st.set_page_config(
        page_title="Task Management System",
        page_icon="✅",
        layout="wide"
    )
    
    st.title("📋 Task Management System")
    st.markdown("Manage your tasks efficiently with this interactive system.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Add Task", "View Tasks", "Edit Task", "Analytics"])
    
    # Load data
    data = load_data()
    tasks = data["tasks"]
    
    if page == "Dashboard":
        st.header("Dashboard")
        
        # Task Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tasks", len(tasks))
        
        completed_tasks = sum(1 for task in tasks if task["status"] == "Completed")
        col2.metric("Completed Tasks", completed_tasks)
        
        pending_tasks = sum(1 for task in tasks if task["status"] == "Pending")
        col3.metric("Pending Tasks", pending_tasks)
        
        high_priority = sum(1 for task in tasks if task["priority"] in ["High", "Critical"] and task["status"] != "Completed")
        col4.metric("High Priority Tasks", high_priority)
        
        # Recent Tasks
        st.subheader("Recent Tasks")
        if tasks:
            sorted_tasks = sorted(tasks, key=lambda x: x["created_at"], reverse=True)[:5]
            for task in sorted_tasks:
                with st.expander(f"{task['title']} ({task['status']})"):
                    st.write(f"**Description:** {task['description']}")
                    st.write(f"**Priority:** {task['priority']}")
                    st.write(f"**Due Date:** {task['due_date']}")
                    st.write(f"**Category:** {task['category']}")
        else:
            st.info("No tasks found. Add some tasks to get started!")
        
        # Quick visualization
        if tasks:
            st.subheader("Task Status Overview")
            fig = create_task_status_visualization(tasks)
            st.pyplot(fig)
    
    elif page == "Add Task":
        st.header("Add New Task")
        
        with st.form("task_form"):
            title = st.text_input("Task Title", max_chars=100)
            description = st.text_area("Description", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
                category = st.text_input("Category (e.g., Work, Personal, Study)")
            
            with col2:
                due_date = st.date_input("Due Date")
            
            submit_button = st.form_submit_button("Add Task")
            
            if submit_button:
                if title and due_date:
                    add_task(
                        title=title,
                        description=description,
                        priority=priority,
                        due_date=due_date.strftime("%Y-%m-%d"),
                        category=category
                    )
                    st.success("Task added successfully!")
                    st.balloons()
                else:
                    st.error("Please fill in all required fields.")
    
    elif page == "View Tasks":
        st.header("View Tasks")
        
        # Filter options
        with st.expander("Filter Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_status = st.multiselect(
                    "Status",
                    ["All", "Pending", "In Progress", "Completed", "Cancelled"],
                    default=["All"]
                )
            
            with col2:
                filter_priority = st.multiselect(
                    "Priority",
                    ["All", "Low", "Medium", "High", "Critical"],
                    default=["All"]
                )
            
            with col3:
                categories = ["All"] + list(set(task["category"] for task in tasks if task["category"]))
                filter_category = st.multiselect(
                    "Category",
                    categories,
                    default=["All"]
                )
        
        # Apply filters
        filtered_tasks = tasks
        
        if "All" not in filter_status:
            filtered_tasks = [task for task in filtered_tasks if task["status"] in filter_status]
        
        if "All" not in filter_priority:
            filtered_tasks = [task for task in filtered_tasks if task["priority"] in filter_priority]
        
        if "All" not in filter_category:
            filtered_tasks = [task for task in filtered_tasks if task["category"] in filter_category]
        
        # Display tasks
        if filtered_tasks:
            for task in filtered_tasks:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    with st.expander(f"{task['title']} ({task['status']})"):
                        st.write(f"**Description:** {task['description']}")
                        st.write(f"**Priority:** {task['priority']}")
                        st.write(f"**Due Date:** {task['due_date']}")
                        st.write(f"**Category:** {task['category']}")
                        st.write(f"**Created At:** {task['created_at']}")
                
                with col2:
                    status_options = ["Pending", "In Progress", "Completed", "Cancelled"]
                    new_status = st.selectbox(
                        "Status",
                        status_options,
                        index=status_options.index(task["status"]),
                        key=f"status_{task['id']}"
                    )
                    
                    if new_status != task["status"]:
                        if update_task_status(task["id"], new_status):
                            st.experimental_rerun()
                    
                    if st.button("Edit", key=f"edit_{task['id']}"):
                        st.session_state.edit_task_id = task["id"]
                        st.experimental_rerun()
                    
                    if st.button("Delete", key=f"delete_{task['id']}"):
                        delete_task(task["id"])
                        st.success("Task deleted!")
                        st.experimental_rerun()
                
                st.markdown("---")
        else:
            st.info("No tasks match the selected filters.")
    
    elif page == "Edit Task":
        st.header("Edit Task")
        
        if hasattr(st.session_state, 'edit_task_id'):
            task_id = st.session_state.edit_task_id
            task = get_task_by_id(task_id)
            
            if task:
                with st.form("edit_task_form"):
                    title = st.text_input("Task Title", value=task["title"], max_chars=100)
                    description = st.text_area("Description", value=task["description"], height=100)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        priority = st.selectbox(
                            "Priority",
                            ["Low", "Medium", "High", "Critical"],
                            index=["Low", "Medium", "High", "Critical"].index(task["priority"])
                        )
                        category = st.text_input("Category", value=task["category"])
                    
                    with col2:
                        # Convert string date to datetime
                        try:
                            task_date = datetime.datetime.strptime(task["due_date"], "%Y-%m-%d").date()
                        except ValueError:
                            task_date = datetime.date.today()
                        
                        due_date = st.date_input("Due Date", value=task_date)
                        status = st.selectbox(
                            "Status",
                            ["Pending", "In Progress", "Completed", "Cancelled"],
                            index=["Pending", "In Progress", "Completed", "Cancelled"].index(task["status"])
                        )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        submit_button = st.form_submit_button("Update Task")
                    
                    with col2:
                        cancel_button = st.form_submit_button("Cancel")
                    
                    if submit_button:
                        if title and due_date:
                            if update_task(
                                task_id=task_id,
                                title=title,
                                description=description,
                                priority=priority,
                                due_date=due_date.strftime("%Y-%m-%d"),
                                category=category,
                                status=status
                            ):
                                st.success("Task updated successfully!")
                                # Clear the edit task ID
                                del st.session_state.edit_task_id
                                st.experimental_rerun()
                        else:
                            st.error("Please fill in all required fields.")
                    
                    if cancel_button:
                        # Clear the edit task ID
                        del st.session_state.edit_task_id
                        st.experimental_rerun()
            else:
                st.error("Task not found!")
                if st.button("Back to Tasks"):
                    # Clear the edit task ID
                    del st.session_state.edit_task_id
                    st.experimental_rerun()
        else:
            st.info("No task selected for editing. Please select a task from the 'View Tasks' page.")
            if st.button("Go to View Tasks"):
                st.experimental_rerun()
    
    elif page == "Analytics":
        st.header("Task Analytics")
        
        if not tasks:
            st.info("No tasks found. Add some tasks to generate analytics.")
        else:
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Status", "Priority", "Category", "Due Dates"])
            
            with tab1:
                st.subheader("Task Status Distribution")
                fig = create_task_status_visualization(tasks)
                st.pyplot(fig)
                
                # Status table
                status_counts = {"Pending": 0, "In Progress": 0, "Completed": 0, "Cancelled": 0}
                for task in tasks:
                    if task["status"] in status_counts:
                        status_counts[task["status"]] += 1
                
                status_df = pd.DataFrame({
                    "Status": status_counts.keys(),
                    "Count": status_counts.values()
                })
                st.dataframe(status_df)
            
            with tab2:
                st.subheader("Task Priority Analysis")
                fig = create_priority_visualization(tasks)
                st.pyplot(fig)
                
                # Priority table with completion rate
                priority_total = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
                priority_completed = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
                
                for task in tasks:
                    priority = task["priority"]
                    if priority in priority_total:
                        priority_total[priority] += 1
                        if task["status"] == "Completed":
                            priority_completed[priority] += 1
                
                priority_df = pd.DataFrame({
                    "Priority": priority_total.keys(),
                    "Total Tasks": priority_total.values(),
                    "Completed": priority_completed.values(),
                    "Completion Rate": [f"{priority_completed[p]/priority_total[p]*100:.1f}%" if priority_total[p] > 0 else "0%" for p in priority_total]
                })
                st.dataframe(priority_df)
            
            with tab3:
                st.subheader("Tasks by Category")
                fig = create_category_visualization(tasks)
                st.pyplot(fig)
                
                # Category table
                categories = {}
                for task in tasks:
                    category = task["category"]
                    if category in categories:
                        categories[category] += 1
                    else:
                        categories[category] = 1
                
                category_df = pd.DataFrame({
                    "Category": categories.keys(),
                    "Count": categories.values()
                })
                st.dataframe(category_df)
            
            with tab4:
                st.subheader("Tasks by Due Date")
                fig = create_due_date_visualization(tasks)
                st.pyplot(fig)
                
                # Upcoming deadlines
                st.subheader("Upcoming Deadlines")
                today = datetime.date.today()
                upcoming_tasks = []
                
                for task in tasks:
                    if task["status"] not in ["Completed", "Cancelled"]:
                        try:
                            due_date = datetime.datetime.strptime(task["due_date"], "%Y-%m-%d").date()
                            days_left = (due_date - today).days
                            if days_left >= 0:
                                upcoming_tasks.append({
                                    "Title": task["title"],
                                    "Due Date": task["due_date"],
                                    "Days Left": days_left,
                                    "Priority": task["priority"]
                                })
                        except ValueError:
                            continue
                
                if upcoming_tasks:
                    upcoming_df = pd.DataFrame(upcoming_tasks)
                    upcoming_df = upcoming_df.sort_values(by="Days Left")
                    st.dataframe(upcoming_df)
                else:
                    st.info("No upcoming deadlines found.")

if __name__ == "__main__":
    main()