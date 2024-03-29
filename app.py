from flask import Flask, request, render_template, jsonify
from flask_mail import Mail, Message
import tensorflow as tf
import cv2, os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from mlxtend.plotting import plot_confusion_matrix

app = Flask(__name__)

prediction = None
precaution = None
per_email, per_name, file_path, image = None, None, None, None
conmat_path, statistics = None, None

mail_username = 'projectosteoporosis@gmail.com'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = mail_username
app.config['MAIL_PASSWORD'] = 'stto ljuy dqyu krwk'
mail = Mail(app)

img_size = 256
model = tf.keras.models.load_model('D:\Classification of Osteoporosis\model.h5')
categories = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
# Using Model
data_path = 'D:\Classification of Osteoporosis\Dataset'
categories_test = os.listdir(data_path)
labels = [i for i in range(len(categories_test))]
label_dict = dict(zip(categories_test, labels))
data = []
label = []

for category in categories_test:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)
    for img_name in img_names:
        #loading every image path of current folder
        img_path = os.path.join(folder_path,img_name)
        #read the image as numpy array
        img = cv2.imread(img_path)   
        try:
            #convert the BGR image to Gray scale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #resize the image into 256x256
            resized = cv2.resize(gray, (img_size, img_size))
            #add the resized image into the data list
            data.append(resized)
            #add the category of current image to label list
            label.append(label_dict[category])            
        except Exception as e:
            #it handles if any exception is occurred
            print('Exception : ',e)
#convert the image to numpy array and normalize the values between 0 and 1
data = np.array(data)/255.0
#reshape the data into a new shape
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
#convert the label list to numpy array
label = np.array(label)
#NumPy array where each row represents a one-hot encoded vector
new_label = to_categorical(label)
#split training and test data
x_train, x_test, y_train, y_test = train_test_split(data, new_label, test_size=0.1)
#Calc Statistics for each image
statistics = {}
#calculate Accuracy
test_labels = np.argmax(y_test, axis=1)
pred_test = model.predict(x_test)
pred_test = np.argmax(pred_test, axis=-1)
cm = confusion_matrix(test_labels, pred_test)
#plot
plt.figure()
plot_confusion_matrix(cm, figsize = (12, 8), hide_ticks = True, cmap = plt.cm.Blues)
plt.xticks(range(5), ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe'], fontsize = 16)
plt.yticks(range(5), ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe'], fontsize = 16)
conmat_path = os.path.join('static/images/confusion_matrix','con_mat.png')
plt.savefig(conmat_path)
#calc statistics of uploaded image
statistics = {
    'Accuracy' : accuracy_score(test_labels, pred_test),
    'Precision' : precision_score(test_labels, pred_test, average = 'macro'),
    'Recall' : recall_score(test_labels, pred_test, average = 'macro'),
    'F1 Score' : f1_score(test_labels, pred_test, average = 'macro'),
    'Confusion Matrix' : cm.tolist()
    }

def predict(image_path):
    #convert to grayscale
    img = Image.open(image_path).convert('L')
    img = img.resize((img_size, img_size))
    img = np.array(img)/255.0 #normalize pixels to 0 | 1
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)   
    #prediction
    pred_single = model.predict(img)
    predict_category = categories[np.argmax(pred_single)]
    return predict_category

def get_precautions(prediction):
    precautions = {
        'Normal' : [" Maintain a balanced and nutritious diet to support bone health.",
                    " Engage in regular weight-bearing excercises like walking, jogging, or dancing to strengthen bones.",
                    " Avoid excessive alcohol consumption and smoking, as they can contribute to bone loss.",
                    " Get regular check-ups and bone density tests as recommended by your healthcare provider."],
        'Doubtful' : [" Consult a healthcare professional for further evaluation and dianosis.",
                      " Follow any additioinal tests or screenings recommended by your heathcare provider.",
                      " Maintain a healthy lifestyle with a focus on nutrition, exercise, and overall well-being."],
        'Moderate' : [" Take necessary precautions to prevent falls, such as removing hazards at home, using assistive devices, and ensuring proper lighting.",
                      " Follow the recommendations of your healthcare provider regarding medication, supplements, and physical therapy.",
                      " Engage in exercises that focus on balance, strength, and flexibility to reduce the risk of fractures.",
                      " Consider modifications in daily activities to prevent excessive strain on the bones."],
        'Mild' : [" Take necessary precautions similar to those for the moderate category.",
                  " Follow the advice of your healthcare provider regarding lifestyle modifications, medication, and therapeutic interventions.",
                  " Engage in exercises that are appropriate for your condition and focus on improving bone strength and flexibility.",
                  " Ensure an adequate intake of calcium, vitamin D, and other essential nutrients for bone health."],
        'Severe' : [" Seek immediate medical attention and follow the guidance of healthcare professionals.",
                    " Adhere strictly to the prescribed treatment plan, including medication, therapy, and lifestyle modifications.",
                    " Take precautions to minimize the risk of falls and fractures, such as using assistive devices and making necessary home modifications.",
                    " Engage in physical activities as recommended by your healthcare provider, considering the limitations of your condition."]
    }
    return precautions.get(prediction)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/send_email', methods=['POST'])
def send_email():
    global per_email, per_name, image, file_path, prediction, precaution, mail_username
    reciepient = per_email
    subject = 'Osteoporosis report'
    message_body = 'Hello '+per_name+',<br>Your report for the X-ray provided : Detected stage is<br>'  
    message = Message(subject=subject,
                      sender=mail_username,
                      recipients=[reciepient],
                      body=message_body)
    with app.open_resource(file_path) as fp:
        message.attach(image.filename, 'image/'+image.filename.split('.')[-1], fp.read())
    bold_text ='<b><i>'+ prediction+'<i><br><u>Precautions :</u></b><br>'
    precautions = ''
    for i in range(len(precaution)):
        precautions+=str(i+1)+'.'+precaution[i]+"<br>"
    last= "<br>Thank you.<br>**This is a computer generated report**"
    html_body = f'{message_body}{bold_text}{precautions}{last}'
    try:
        message.html = html_body
        mail.send(message)
        return jsonify({"message":"Email sent successfully."})
    except Exception as e:
        return jsonify({"message":"Error in sending mail."})

@app.route('/upload', methods=['POST'])
def upload_file():
    global prediction, precaution, file_path, image, per_email, per_name
    per_email = request.form['pmail']
    per_name = request.form['pname']
    if 'image' not in request.files:
        return 'No file part'
    image = request.files['image']
    if image.filename == '':
        return 'No selected file'
    file_path = os.path.join('static/images/uploaded_images', image.filename)
    image.save(file_path) 
    prediction = predict(file_path)
    precaution = get_precautions(prediction)
    return render_template('result.html', image_path=file_path, image_name=image.filename, prediction = prediction, precaution = precaution)

@app.route('/statistic')
def statistic():
    global conmat_path, statistics
    return render_template('statistic.html', conmat_path = conmat_path, statistics = statistics)

if __name__ == "__main__":
    app.run(debug=False)