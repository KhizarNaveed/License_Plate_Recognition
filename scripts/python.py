from roboflow import Roboflow

# Initialize with your API key
rf = Roboflow(api_key="UwAY8VBabbc7tsIghppH")

# Print available workspaces to see what you have access to
print("Workspace info:", rf.workspace())

try:
    # Try to access a public project directly
    # The format appears to be workspace-name/project-name
    project = rf.workspace("hassan-xjtvy").project("weapon-detection")
    print("Found project:", project)
    
    # Get available versions
    versions = project.versions()
    print("Available versions:", versions)
    
    # Download the latest version in YOLOv8 format
    dataset = project.version(1).download("yolov8")
    print("Dataset downloaded successfully to:", dataset)
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")
    
    # If the above fails, try searching for the project
    try:
        # Try using your workspace to search for all projects
        my_workspace = rf.workspace()
        projects = my_workspace.project_list()
        print("Your projects:", projects)
        
        # Look for any weapon detection projects
        weapon_projects = [p for p in projects if "weapon" in p.lower()]
        if weapon_projects:
            print("Found weapon-related projects:", weapon_projects)
    except Exception as e2:
        print(f"Second approach failed: {e2}")
#spectral
#Cnn
#RNN