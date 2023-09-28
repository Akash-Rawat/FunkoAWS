from io import BytesIO
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import gradio as gr

# Combined Code for Beard and Hairstyle Detection and Styling

background_image_paths = [
    "Data/AdobeColorFunko/Outfits/DummyDress1.png",
    "Data/AdobeColorFunko/Outfits/GlassesDummy.png",
    "Data/AdobeColorFunko/Outfits/DummyDress3.png"
]


class GenderClassifier:
    def __init__(self, model_path, class_names):
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_gender(self, image_path):
        input_image = self.preprocess_image(image_path)

        with torch.no_grad():
            predictions = self.model(input_image)

        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]

        return predicted_label
    
class WomenHairStyleClassifier:
    def __init__(self, model_path, class_names):
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image
    
    def load_model(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_hairStyle(self, image_path):
        input_image = self.preprocess_image(image_path)

        with torch.no_grad():
            predictions = self.model(input_image)

        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]

        return predicted_label
    
class WomenHairColorClassifier:
    def __init__(self, model_path, class_names):
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_hairColor(self, image_path):
        input_image = self.preprocess_image(image_path)

        with torch.no_grad():
            predictions = self.model(input_image)

        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]

        return predicted_label
# Function to classify beard style
class BeardClassifier:
    def __init__(self, model_path, class_names):
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image):
        image = Image.open(image).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_beard(self, image):
        input_image = self.preprocess_image(image)
        with torch.no_grad():
            predictions = self.model(input_image)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]
        return predicted_label

# Function to classify beard color
class BeardColorClassifier:
    def __init__(self, model_path, class_names):
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image):
        image = Image.open(image).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_beard_color(self, image):
        input_image = self.preprocess_image(image)
        with torch.no_grad():
            predictions = self.model(input_image)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]
        return predicted_label


# Function to classify hairstyle
class HairStyleClassifier:
    def __init__(self, model_path, class_names):
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image):
        image = Image.open(image).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_hair(self, image):
        input_image = self.preprocess_image(image)
        with torch.no_grad():
            predictions = self.model(input_image)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]
        return predicted_label

class MenHairColorClassifier:
    def __init__(self, model_path, class_names):
        self.model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(class_names))
        self.load_model(model_path)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_names = class_names

    def preprocess_image(self, image):
        image = Image.open(image).convert("RGB")
        image = self.data_transforms(image)
        image = image.unsqueeze(0)
        return image

    def load_model(self, model_path):
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def classify_menHair_color(self, image):
        input_image = self.preprocess_image(image)
        with torch.no_grad():
            predictions = self.model(input_image)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = self.class_names[predicted_class]
        return predicted_label


def dummy_eye(background_image, x, y, placeholder_image_path, x_coordinate, y_coordinate):
    placeholder_image = Image.open(placeholder_image_path)
    target_size = (x, y)
    placeholder_image = placeholder_image.resize(target_size, Image.LANCZOS)
    placeholder_array = np.array(placeholder_image)
    placeholder_width, placeholder_height = placeholder_image.size
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    background_image.paste(placeholder_image, region_box, mask=placeholder_mask)
    background_array = np.array(background_image)

# Function to overlay a beard on a background image
def process_image_Beard(background_image, x, placeholder_image_path, x_coordinate, y_coordinate):
    placeholder_image = Image.open(placeholder_image_path)
    target_size = (x, x)
    placeholder_image = placeholder_image.resize(target_size, Image.LANCZOS)
    placeholder_array = np.array(placeholder_image)
    placeholder_width, placeholder_height = placeholder_image.size
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    background_image.paste(placeholder_image, region_box, mask=placeholder_mask)
    background_array = np.array(background_image)
    placeholder_alpha = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None

def process_image_WomanHair(background_image, x, y, placeholder_image_path, x_coordinate, y_coordinate):
    placeholder_image = Image.open(placeholder_image_path)
    target_size = (x, y)
    placeholder_image = placeholder_image.resize(target_size, Image.LANCZOS)
    placeholder_array = np.array(placeholder_image)
    placeholder_width, placeholder_height = placeholder_image.size
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    background_image.paste(placeholder_image, region_box, mask=placeholder_mask)
    background_array = np.array(background_image)
    placeholder_alpha = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None


def add_eyebrow(background_image, x_coordinate, y_coordinate, eyebrow_image_path):
    eyebrow_image = Image.open(eyebrow_image_path)
    target_size = (200, 200)  # Adjust the size as needed
    eyebrow_image = eyebrow_image.resize(target_size, Image.LANCZOS)
    region_box = (x_coordinate, y_coordinate, x_coordinate + eyebrow_image.width, y_coordinate + eyebrow_image.height)
    eyebrow_mask = eyebrow_image.split()[3] if eyebrow_image.mode == 'RGBA' else None
    background_image.paste(eyebrow_image, region_box, mask=eyebrow_mask)
    background_array = np.array(background_image)


    
    
# Function to overlay a hairstyle on a background image
def process_image_menHair(background_image, x, y, placeholder_image_path, x_coordinate, y_coordinate):
    placeholder_image = Image.open(placeholder_image_path)
    target_size = (x, y)
    placeholder_image = placeholder_image.resize(target_size, Image.LANCZOS)
    placeholder_array = np.array(placeholder_image)
    placeholder_width, placeholder_height = placeholder_image.size
    region_box = (x_coordinate, y_coordinate, x_coordinate + placeholder_width, y_coordinate + placeholder_height)
    placeholder_mask = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None
    background_image.paste(placeholder_image, region_box, mask=placeholder_mask)
    background_array = np.array(background_image)
    placeholder_alpha = placeholder_image.split()[3] if placeholder_image.mode == 'RGBA' else None

# Function to generate Funko figurines
def generate_funko_figurines(input_image):

    WomenHairStyle_classifier = WomenHairStyleClassifier('Data/FunkoSavedModels/WomenHairStyle.pt', ['MediumLength', 'ShortHair', 'SidePlait'])
    predicted_WomenHairStyle = WomenHairStyle_classifier.classify_hairStyle(input_image)

    WomenHairColor_classifier = WomenHairColorClassifier('Data/FunkoSavedModels/WomenHairColor.pt', ['Black', 'Brown', 'Ginger', 'White'])
    predicted_WomenHairColor = WomenHairColor_classifier.classify_hairColor(input_image)
    # Detect and classify gender
    gender_classifier = GenderClassifier('Data/FunkoSavedModels/Gender.pt', ['Female', 'Male'])
    predicted_gender = gender_classifier.classify_gender(input_image)

    # Detect and classify beard style
    beard_classifier = BeardClassifier('Data/FunkoSavedModels/FunkoResnet18BeardStyle.pt', ['Bandholz', 'CleanShave', 'FullGoatee', 'Moustache', 'RapIndustryStandards', 'ShortBeard'])
    predicted_style_label = beard_classifier.classify_beard(input_image)

    # Detect and classify beard color
    beard_color_classifier = BeardColorClassifier('Data/FunkoSavedModels/FunkoResnet18BeardColor.pt', ['Black', 'DarkBrown', 'Ginger', 'LightBrown', 'SaltAndPepper', 'White'])
    predicted_color_label = beard_color_classifier.classify_beard_color(input_image)

    # Classify hairstyle
    hair_style_classifier = HairStyleClassifier('Data/FunkoSavedModels/FunkoResnet18HairStyle.pt', ['Afro', 'Bald', 'Puff', 'Spike'])
    predicted_hairStyle_label = hair_style_classifier.classify_hair(input_image)

    #classify menHairColor
    menhair_color_classifier = MenHairColorClassifier('Data/FunkoSavedModels/FunkoResnet18MenHairColor.pt', ['Black', 'DarkBrown', 'Ginger', 'LightBrown', 'SaltAndPepper', 'White'])
    predicted_menhairColor_label = menhair_color_classifier.classify_menHair_color(input_image)
    # Process background images and apply beard style and color along with hair style and color
    final_images = []

    for background_image_path in background_image_paths:
        background_image = Image.open(background_image_path)
        x_coordinate = 90
        y_coordinate = 50
        add_eyebrow(background_image, 115, 80, "Data/AdobeColorFunko/EyezBrowz/Eyebrow.png")
        #dummy_eye(background_image, 245, 345, 'Data/AdobeColorFunko/EyezBrowz/MaleEye.png', x_coordinate, y_coordinate)
        if predicted_gender == 'Male':
            x = 245
            y = 345
            placeholder_image_path = f"Data/AdobeColorFunko/EyezBrowz/{predicted_gender}Eye.png"
            x_coordinate = 90
            y_coordinate = 50
            dummy_eye(background_image, x, y, placeholder_image_path, x_coordinate, y_coordinate)

            if predicted_style_label == 'Bandholz':
                process_image_Beard(background_image, 320,
                                     f"Data/AdobeColorFunko/Beard/Bandholz/{predicted_color_label}.png",
                                     50, 142)

            if predicted_style_label == 'ShortBeard':
                process_image_Beard(background_image, 300,
                                     f"Data/AdobeColorFunko/Beard/ShortBeard/{predicted_color_label}.png",
                                     62, 118)

            if predicted_style_label == 'FullGoatee':
                process_image_Beard(background_image, 230,
                                     f"Data/AdobeColorFunko/Beard/Goatee/{predicted_color_label}.png",
                                     96, 168)

            if predicted_style_label == 'RapIndustryStandards':
                process_image_Beard(background_image, 290,
                                     f"Data/AdobeColorFunko/Beard/RapIndustry/{predicted_color_label}.png",
                                     67, 120)

            if predicted_style_label == 'Moustache':
                process_image_Beard(background_image, 220,
                                     f"Data/AdobeColorFunko/Beard/Moustache/{predicted_color_label}.png",
                                     100, 160)

            if predicted_style_label == 'CleanShave':
                process_image_Beard(background_image, 220,
                                     f"Data/AdobeColorFunko/Beard/CleanShave/{predicted_color_label}.png",
                                     100, 160)

            # Add other conditions for different beard styles

            # Overlay hairstyle
            if predicted_hairStyle_label == 'Afro':
                process_image_menHair(background_image, 336, 420,
                                       f"Data/AdobeColorFunko/MenHairstyle/Afro/{predicted_menhairColor_label}.png",
                                       41, 76)

            if predicted_hairStyle_label == 'Puff':
                process_image_menHair(background_image, 320, 420,
                                       f"Data/AdobeColorFunko/MenHairstyle/Puff/{predicted_menhairColor_label}.png",
                                       50, 68)

            if predicted_hairStyle_label == 'Spike':
                process_image_menHair(background_image, 310, 420,
                                       f"Data/AdobeColorFunko/MenHairstyle/Spike/{predicted_menhairColor_label}.png",
                                       50, 70)

            if predicted_hairStyle_label == 'Bald':
                process_image_menHair(background_image, 310, 420,
                                       f"Data/AdobeColorFunko/MenHairstyle/Bald/{predicted_menhairColor_label}.png",
                                       67, 120)


        if predicted_gender == 'Female':
            x = 245
            y = 345
            placeholder_image_path = f"Data/AdobeColorFunko/EyezBrowz/{predicted_gender}Eye.png"
            x_coordinate = 90
            y_coordinate = 50
            dummy_eye(background_image, x, y, placeholder_image_path, x_coordinate, y_coordinate)
            if predicted_WomenHairStyle == 'MediumLength':
                process_image_WomanHair(background_image, 300,460,
                                     f"Data/AdobeColorFunko/WomenHairstyle/MediumLength/{predicted_WomenHairColor}.png",
                                     56, 50)

            if predicted_WomenHairStyle == 'ShortHair':
                process_image_WomanHair(background_image, 270,460,
                                     f"Data/AdobeColorFunko/WomenHairstyle/ShortHair/{predicted_WomenHairColor}.png",
                                     58, 52)

            if predicted_WomenHairStyle == 'SidePlait':
                process_image_WomanHair(background_image, 300,450,
                                     f"Data/AdobeColorFunko/WomenHairstyle/SidePlait/{predicted_WomenHairColor}.png",
                                     54, 56)


        # Convert the resulting image to base64
        buffered = BytesIO()
        background_image.save(buffered, format="PNG")
        #base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        final_images.append(background_image)

    return final_images
imageComponent = gr.Image(type="filepath")

# Define Gradio input components
input_image = gr.inputs.Image(type="pil", label="Upload your image")
background_images = [gr.inputs.Image(type="pil", label="Background Image " + str(i + 1)) for i in range(3)]

# Create Gradio interface
gr.Interface(
    fn=generate_funko_figurines,
    inputs=imageComponent,
    outputs=[gr.outputs.Image(type="pil", label="Generated Image " + str(i + 1)) for i in range(3)],
    title="Funko Figurine Generator",
    description="Generate personalized Funko figurines with different styles and backgrounds.",
).launch()
