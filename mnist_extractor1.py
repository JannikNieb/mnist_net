import numpy as NP
import scipy



class MNISTExtractor:
    
    def __init__(self):
        return
        
        
    def extractImages(self, filename, custom_image_count):
        
        with open(filename, 'rb') as file:
            secureCode = NP.fromfile(file, dtype='>u4', count=1)
            
            if secureCode != 0x0803:
                print("Invalid image file!")
                file.close()
                return
                
            imageCount = NP.fromfile(file, dtype='>u4', count=1)[0]
            width = NP.fromfile(file, dtype='>u4', count=1)[0]
            height = NP.fromfile(file, dtype='>u4', count=1)[0]
            pixels = width * height

            image_data = []
            
            for i in range(custom_image_count):
                image_data.append(NP.asfarray(NP.fromfile(file, NP.uint8, count=pixels) / 255.0 * 0.99) + 0.01)

            return image_data
               

    def extractLabels(self, filename):
        
        with open(filename) as file:
            secureCode = NP.fromfile(file, dtype='>u4', count=1)
            
            if secureCode != 0x0801:
                print("Invalid label file!")
                file.close()
                return
                
            imageCount = NP.fromfile(file, dtype='>u4', count=1)[0]
            label_data = NP.fromfile(file, dtype=NP.uint8, count=imageCount)

            return label_data