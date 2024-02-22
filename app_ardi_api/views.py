from django.shortcuts import render 
from django.http import HttpResponse, HttpResponseBadRequest 
from rest_framework.renderers import JSONRenderer
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from . models import ImageData
from . serializers import ImageDataSerializer

import json, os, cv2, math, uuid, requests
from PIL import Image 
from io import BytesIO 
import numpy as np  
import matplotlib.pyplot as plt   
from rest_framework.response import Response

# Create your views here.

def get_direction_vector(x1, y1, x2, y2):

    point1 = np.array([x1, y1]) 
    point2 = np.array([x2, y2])
    direction_vector = point2 - point1 

    return direction_vector 

def rotate_vector_90(divided_vector):
    x, y = divided_vector
    rotated_x = -y
    rotated_y = x
    return [rotated_x, rotated_y]

def find_direction_vector(data): 

    ##############################################################
    # Task 1 Finding direction vector 
    ##############################################################
    direction_vectors = []

    for xy in range(len(data)-1):
        data_p = data[xy][0]  
        data_p2 = data[xy+1][0] 
        x1, y1 = (data_p[0] / 100) / 10, (data_p[1]/ 100) / 10 
        x2, y2 = (data_p2[0] / 100) / 10, (data_p2[1] /100) / 10 
 
        direction_vector = get_direction_vector(x1, y1, x2, y2) 
        direction_vectors.append(direction_vector)  
    

    data_l = data[len(data)-1]
    data_f = data[0] 
    x1, y1 = (data_l[0][0] / 100) / 10, (data_l[0][1]/ 100) / 10
    x2, y2 = (data_f[0][0] / 100) / 10, (data_f[0][1] /100) / 10 
    
    last_direction_vector = get_direction_vector(x1, y1, x2, y2)
    direction_vectors.append(last_direction_vector)

    ###############################################################
    # Task 2 Finding magnitude
    ###############################################################
    magnitudes = []

    for points in direction_vectors: 
        x = points[0] 
        y = points[1] 
        magnitude = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
        magnitudes.append(magnitude) 

    #find maximum magnitude 
    maximum_magnitude =  max(magnitudes)  

    # Vector for maximum magnitude 
    index = np.argmax(magnitudes) 
    vector_of_maximum_magnitude = direction_vectors[index] 


    # Dividing vector with maximum magnitude 
    divided_vector = np.divide(vector_of_maximum_magnitude, maximum_magnitude)

    # Rotate vector 90 degree 
    rotated_vector = rotate_vector_90(divided_vector)

    # Swap y and z value and keep y=0.0
    x = rotated_vector[0] 
    y = 0.0 
    z = rotated_vector[1] 
    directions = np.array([x, y, z])
    return directions

def prepare_room_data(approx):
    
    normalizedPositions= [] 
    directions = []

    for pts in approx:
        x_y_coordinates = pts[0] 
        x_pts = (x_y_coordinates[0] / 100.0) / 10 
        y_pts = (x_y_coordinates[1] / 100.0) / 10
        
        # x, y, z, normalized(x, y, z, magnitude, sqrMagnitude), magnitude, sqrMagnitude 
        magnitude = math.sqrt(math.pow(x_pts, 2) + math.pow(y_pts, 2)) 
        sqrMagnitude = np.square(magnitude)

        normalized = {}
        normalized['x'] = x_pts/magnitude
        normalized['y'] = y_pts/magnitude 
        normalized['z'] = 0.0  
        normalized['magnitude'] = 1.0 
        normalized['sqrMagnitude'] = 1.0    
         
        normalizedPositions.append({'x': x_pts, 'y': y_pts, 'z': 0.0, 'normalized': normalized, 'magnitude': magnitude, 'sqrMagnitude': sqrMagnitude})   
        # x, y, z, magnitude, sqrMagnitude  
        directions.append({'x': 0.0, 'y': 0.0, 'z': 0.0, 'magnitude': 0.0, 'sqrMagnitude': 0.0}) 
    all_data = {"type": "Room", "name": "",  "width": 1.0, "height": 1.0, "ypos": 1.0, "hasWindow": False,  "windowFrameSize": 0.0, "windowSizeH": 0.0, "windowSizeV": 0.0, "windowSubDivH": 0, "windowSubDivV": 0, "normalizedPositions": normalizedPositions, "directions": directions}
    
    return all_data   


def prepare_window_data(approx): 

    normalizedPositions= [] 
    directions = []
    points = [] 

    for pts in approx:
        x_y_coordinates = pts[0] 
        points.append(x_y_coordinates) 

    # convert to array 
    points_array = np.array(points) 
    # Calculate center point coordinates
    center_x = (np.mean(points_array[:, 0]) /100) /10
    center_y = (np.mean(points_array[:, 1]) /100) /10 

    # x, y, z, normalized(x, y, z, magnitude, sqrMagnitude), magnitude, sqrMagnitude 
    magnitude = math.sqrt(math.pow(center_x, 2) + math.pow(center_y, 2)) 
    sqrMagnitude = np.square(magnitude)

    normalized = {}
    normalized['x'] = center_x/magnitude
    normalized['y'] = center_y/magnitude 
    normalized['z'] = 0.0  
    normalized['magnitude'] = 1.0 
    normalized['sqrMagnitude'] = 1.0

    normalizedPositions.append({'x': center_x, 'y': center_y, 'z': 0.0, 'normalized': normalized, 'magnitude': magnitude, 'sqrMagnitude': sqrMagnitude})   

    #  x, y, z, magnitude, sqrMagnitude
    directions_points = find_direction_vector(approx)  
    directions.append({'x': directions_points[0], 'y': directions_points[1], 'z': directions_points[2], 'magnitude': 1.0, 'sqrMagnitude': 1.0}) 
    
    all_data = {"type": "Window", "name": "",  "width": 1.0, "height": 1.0, "ypos": 1.0, "hasWindow": False,  "windowFrameSize": 0.5, "windowSizeH": 1.0, "windowSizeV": 0.441, "windowSubDivH": 2, "windowSubDivV": 1, "normalizedPositions": normalizedPositions, "directions": directions}
    return all_data 

def prepare_door_data(approx): 

    normalizedPositions= [] 
    directions = []

    for pts in approx:
        x_y_coordinates = pts[0] 
        x_pts = (x_y_coordinates[0] / 100.0) / 10 
        y_pts = (x_y_coordinates[1] / 100.0) / 10
        break
 
    # x, y, z, normalized(x, y, z, magnitude, sqrMagnitude), magnitude, sqrMagnitude 
    magnitude = math.sqrt(math.pow(x_pts, 2) + math.pow(y_pts, 2)) 
    sqrMagnitude = np.square(magnitude)

    normalized = {}
    normalized['x'] = x_pts/magnitude
    normalized['y'] = x_pts/magnitude 
    normalized['z'] = 0.0  
    normalized['magnitude'] = 1.0 
    normalized['sqrMagnitude'] = 1.0


    normalizedPositions.append({'x': x_pts, 'y': y_pts, 'z': 0.0, 'normalized': normalized, 'magnitude': magnitude, 'sqrMagnitude': sqrMagnitude})   

    # x, y, z, magnitude, sqrMagnitude  
    directions_points = find_direction_vector(approx) 
    directions.append({'x': directions_points[0], 'y': directions_points[1], 'z': directions_points[2], 'magnitude': 1.0, 'sqrMagnitude': 1.0}) 

    all_data = {"type": "Door", "name": "",  "width": 1.0, "height": 1.0, "ypos": 0.0, "hasWindow": False,  "windowFrameSize": 0.05, "windowSizeH": 1.0, "windowSizeV": 0.441, "windowSubDivH": 2, "windowSubDivV": 1, "normalizedPositions": normalizedPositions, "directions": directions}
    return all_data 

def contour_detection_and_json_preparation(img):
    # Reading image, convert to gray scale, finding threshold
    image = cv2.imread(img) 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray scale
    #ret, im = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY_INV) # finding threshold 
    img_blur = cv2.medianBlur(image_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11, 2)

    # Finding contour, and filtering and drop unnecessary contour
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    min_area = 100  # Adjust this threshold based on your image and requirements
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area] # filtering contour
    large_contours.pop() 
    #print(len(large_contours))

    # Create a mask to store the segmented regions
    segmented_image = np.zeros_like(image) 

    rooms = []

    # Iterate through the contours
    for i, contour in enumerate(large_contours):
        
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        #print("area is :", area)

        # finding Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        #print("approx values are :", approx) 

        # reverse printing of approx 
        approx[:] = approx[::-1] 
        #print("Reverse approx is: ", approx) 

        # counting approx values
        one_contour_total_points = len(approx) 
        #print("Total points in contour: ", one_contour_total_points) 

        # preparing coordinates for biggest area like room 
        if area > 10000:
            one_room = prepare_room_data(approx)
            rooms.append(one_room) 
        # window detection 
        elif one_contour_total_points == 4 and area < 8000 and area > 2000 : 
            one_window = prepare_window_data(approx) 
            rooms.append(one_window) 
        # door detection 
        elif one_contour_total_points == 3 and area < 8000: 
            one_door = prepare_door_data(approx) 
            rooms.append(one_door) 
        else: 
            print("Exception area: ", area) 

            
        # Draw the contour on the segmented image
        colors = [(0, 105, 255), (255, 0, 0), (255, 150, 0), (50, 0, 255), (220, 255, 0), (255, 255, 0), (0, 255, 255), (0, 255, 0), (255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0), (0, 0, 255)]
        cv2.drawContours(segmented_image, [contour], -1, colors[i], thickness=cv2.FILLED)
    
    #print("Number of rooms: ", len(rooms)) 
    


    # preparing details of json file
    floors = []
    spaces = []
    settings = {
    "wallsHeight": 3.0,
    "doorsHeight": 2.5,
    "interiorWallThickness": 0.05, 
    "exteriorWallThickness": 0.1,
    "windowsThickness": 0.06,
    "doorsThickness": 0.06,
    "roofThickness": 0.2,
    "roofOverhang": 0.2,
    "roofType": 2
    }

    spaces.append(rooms)  
    floors.append({"uniqueId": str(uuid.uuid1()), "spaces": spaces[0]}) 
    final_dict = {"version": "v2", "floors": floors, 'settings': settings}   

    return final_dict, segmented_image 



@csrf_exempt
def input_image(request): 

    if request.method == "POST": 
        #################################################################
        # Reading Image from URL 
        #################################################################
        try: 
            # URL = "/home/hasan/SoftNursery/Ardi/ardi_api/two_room.jpeg"
            # response_img = requests.get(url=URL) 
            # response_img.raise_for_status()

            img_file = request.FILES['image'] 

            # Save the image in a folder 
            original_img_path = os.path.join(settings.STATIC_DIR, "original_img") 
            # clean the directory 
            all_path = [original_img_path]
            for fol in all_path: 
                all_file = os.listdir(fol) 
                for file_ in all_file: 
                    os.remove(os.path.join(fol, file_)) 
                    #print("folder is cleaned") 

            # save the image 
            destination = os.path.join(settings.STATIC_DIR, "original_img", "original.png") 
            with open(destination, 'wb+') as destination_file: 
                for chunk in img_file.chunks(): 
                    destination_file.write(chunk) 

            #reading image from saved directory 
            img_path = os.path.join(settings.STATIC_DIR, 'original_img', 'original.png') 

            # finding room, window, door points 
            data, seg_img = contour_detection_and_json_preparation(img_path)  
            data = json.dumps(data, indent=4)  
            
            # Saving Data to the Model
            model_instance = ImageData(img_dict=data) 
            model_instance.save()  

            ###########################################################################
            # Reading Data from the Model and sending in web
            ###########################################################################
            # complex data 
            last_object = ImageData.objects.last() 
            json_data = last_object.img_dict if last_object else None
            try: 
                last_json_data = json.dumps(json_data)
                last_json_data = json_data
            except KeyError:
                # Handle the case where the JSON data is an empty list 
                last_json_data = None 
            except json.JSONDecodeError:
                # Handle the case where the JSON data is invalid
                last_json_data = None
            
            #json.dump(data, file, indent=4)  
            return HttpResponse(last_json_data) 
            #return last_json_data 
            #return HttpResponse('Image received and processed successfully!')
            #return Response(last_json_data) 

        except: 
            #print("Error! Something wrong.") 
            return HttpResponseBadRequest('Image is not Read! Something Wrong') 
    
    else:
    #     #return render(request, 'index.html') 
        #return HttpResponseBadRequest("Only POST and GET requests are allowed") 
        return HttpResponse("Only POST and GET requests are allowed") 
        #return print("console print") 
        




# def getting_json_data(request):

#     if request.method == "GET": 
#         ###########################################################################
#         # Reading Data from the Model and sending in web
#         ###########################################################################
#         # complex data 
#         last_object = ImageData.objects.last() 
#         json_data = last_object.img_dict if last_object else None
#         try:
#             last_json_data = json.dumps(json_data)
#             last_json_data = json_data
#         except KeyError:
#             # Handle the case where the JSON data is an empty list 
#             last_json_data = None 
#         except json.JSONDecodeError:
#             # Handle the case where the JSON data is invalid
#             last_json_data = None
        
#         #json.dump(data, file, indent=4) 
#         return HttpResponse(last_json_data)

        
