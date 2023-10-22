
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import os #python package/path connectivity/define path
from skimage.io import imread #to read a image
from skimage.transform import resize #to resize the image


data_folder = "C:\\Users\\fadil\\Downloads\\Maskdata-20231021T115547Z-001\\Maskdata"



# To get the list of directories

os.listdir("C:\\Users\\fadil\\Downloads\\Maskdata-20231021T115547Z-001\\Maskdata")

# To get count of files in a directory
len(os.listdir("C:\\Users\\fadil\\Downloads\\Maskdata-20231021T115547Z-001\\Maskdata\\with_mask"))


# To get count of files in a directory
len(os.listdir("C:\\Users\\fadil\\Downloads\\Maskdata-20231021T115547Z-001\\Maskdata\\without_mask"))


maskpath=os.path.join("C:\\Users\\fadil\\Downloads\\Maskdata-20231021T115547Z-001\\Maskdata\\with_mask")
for img in os.listdir(maskpath):
  print(img)

withoutmaskpath=os.path.join("C:\\Users\\fadil\\Downloads\\Maskdata-20231021T115547Z-001\\Maskdata\\without_mask")
for img in os.listdir(withoutmaskpath):
  print(img)

flat_data_arr=[]  #for input data
target_arr=[]     #output data
categories=['with_mask','without_mask']    #with mask==0  without mask==1

datapath="C:\\Users\\fadil\\Downloads\\Maskdata-20231021T115547Z-001\\Maskdata"
for i in categories:    #Mask #without_mask
  print('data is loading.....')
  path=os.path.join(datapath,i)   #/content/drive/MyDrive/project data cnn/Maskdata/maskdata'   #'/content/drive/MyDrive/project data cnn/Maskdata/without_mask'
  for img in os.listdir(path):
    img_arr=imread(os.path.join(path,img))      #'/content/drive/MyDrive/project data cnn/Data/Cat/img1'.....img2....img3
    img_resize=resize(img_arr,(150,150,3))  #RGB  # all image different size, so we need to resize to same size
    flat_data_arr.append(img_resize.flatten())
    target_arr.append(categories.index(i))
  print('Data Uploading Completed....',i)

flat_data=np.array(flat_data_arr)     # Converted into array
target_data=np.array(target_arr)      # Converted into array
df=pd.DataFrame(flat_data)
df['Target']=target_data
df

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix,accuracy_score
print('Accuracy Score is....',accuracy_score(y_test,y_pred))

#********************************************************************
#********************************************************************
#********************************************************************
#********************************************************************


import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# Create a function to open a file dialog for selecting a JPG file
def open_file_dialog():
    global a,answer,mask_label
    file_path = filedialog.askopenfilename(filetypes=[("JPG files", "*.jpg")])
    img = imread(file_path)
    img = resize(img, (150, 150, 3)).flatten().reshape(1, -1)
    a = model.predict(img)

    if file_path:
        load_and_display_image(file_path)

# Create a function to load and display the selected image
def load_and_display_image(file_path):
    global photo
    image = Image.open(file_path)
    image.thumbnail((300, 300))  # Resize the image to fit in the window
    photo = ImageTk.PhotoImage(image)

    # Display the image on the GUI
    image_label.config(image=photo)
    image_label.photo = photo

    if a == 0:
         mask_label.config(text="Has Mask")      # Create a label to display
    else:
        mask_label.config(text="No Mask")       # Create a label to display




# Create the main application window
app = tk.Tk()
app.title("Face Mask Detection")

# Create the result label
mask_label = tk.Label(app, text="",font=('Helvetica', 14, 'bold'))
mask_label.pack()

# Set the window size to be larger
app.geometry("800x600")

# Create a style for the button
style = ttk.Style()
style.configure("TButton", padding=(10, 10, 10, 10), font=("Helvetica", 14))


# Create and configure a label to display the uploaded image
image_label = ttk.Label(app)
image_label.pack()



# Create an "Upload JPG" button with styling
upload_button = ttk.Button(app, text="Upload Image", command=open_file_dialog,)
upload_button.pack(pady=20)



# Run the GUI application
app.mainloop()