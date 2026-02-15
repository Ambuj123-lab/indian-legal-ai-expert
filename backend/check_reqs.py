def check_requirements():
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    if "en_core_web_sm" in content:
        print("✅ en_core_web_sm is already in requirements.txt")
    else:
        print("❌ en_core_web_sm NOT found in requirements.txt. Adding it is recommended.")

    print(f"File size: {len(content)} bytes")

check_requirements()
