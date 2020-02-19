import os 
classes = ["mask","nomask"]
with open('./data/train.txt','w') as f:
    after_generate = os.listdir("./data/image/train")
    for image in after_generate:
    	if image.endswith("jpg"):
        	f.write(image + ";" + str(classes.index(image.split("_")[0]))+ "\n")
