import requests

#get_response = requests.get("http://localhost:8080/hello")
#get_content = get_response.content
#print(get_content)

#post_response = requests.post("http://localhost:8080/hello_dynamic", data = {"name": "Leora"}) #ERROR 500: Missing name
#post_response = requests.post("http://localhost:8080/hello_dynamic", data = "Leora") #ERROR 500: missing name
post_response = requests.post("http://localhost:8080/hello_dynamic", "Leora") #ERROR 500: missing name 
post_content = post_response.content
print(post_content)
