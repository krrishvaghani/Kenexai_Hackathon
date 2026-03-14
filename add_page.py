import sys

filepath = 'D:/kenexai_round2/Kenexai_Hackathon/dashboard/app.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Modify Sidebar
old_sidebar = 'page = st.sidebar.radio("Go to Page:", ["Data Quality Monitor", "Exploratory Data Analysis", "Business Personas", "Policy Risk Prediction Tool"])'
new_sidebar = 'page = st.sidebar.radio("Go to Page:", ["Data Quality Monitor", "Exploratory Data Analysis", "Business Personas", "Policy Risk Prediction Tool", "Insurance AI Copilot"])'

if old_sidebar in content:
    content = content.replace(old_sidebar, new_sidebar)
    
    # Append routing logic
    page_routing = '''
# -------------------------------------------------------------------
# PAGE 5: INSURANCE AI COPILOT
# -------------------------------------------------------------------
elif page == "Insurance AI Copilot":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from genai.copilot_chatbot_page import render_copilot_page
    render_copilot_page()
'''
    content += page_routing
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("UI page appended!")
else:
    print("Could not find sidebar string.")
