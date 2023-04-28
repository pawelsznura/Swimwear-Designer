import os 

# get number of generated img 
# Get a list of all files in the folder
files = os.listdir("created_images_text")

# Get the number of files in the folder
num_files = len(files)

for x in range(1,num_files):
    f = open("created_images_text/"+str(x)+".txt", "r")
    text = f.read()
    lines = text.split("\n")
        # Loop over the lines and limit the number of characters in the line starting with "prompt:"
    for i, line in enumerate(lines):
        if line.startswith("prompt:"):
            print(str(x)+" " + line)
    
