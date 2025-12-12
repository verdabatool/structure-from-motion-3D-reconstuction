import xmltodict
import json

with open("project_data.xml", "r") as f:
    xml_data = f.read()

# XML → dict
data = xmltodict.parse(xml_data)

# dict → formatted JSON string
json_str = json.dumps(data, indent=4)

# Save to file
with open("cameras.json", "w") as f:
    f.write(json_str)

print("Saved as output.json")
