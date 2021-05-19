import os
import sys
import pdb  
import code
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image 

# Ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Camera info
viewPoint=np.array([0,0,0]).astype(np.float)
viewDir=np.array([0,0,-1]).astype(np.float)
viewUp=np.array([0,1,0]).astype(np.float)
viewRight=np.array([1,0,0]).astype(np.float)
viewProjNormal=-1*viewDir
viewWidth=1.0
viewHeight=1.0
projDistance=1.0

# Colors
emptyColor=np.array([0,0,0]).astype(np.float)
ambientColor=np.array([0,0,0]).astype(np.float)

class Color:
    def __init__(self, rgb):
        self.color=rgb

    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma;
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)

class Shader:
    def __init__(self, diffuse_color, specular_color = None, exponent = None):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.exponent = exponent 

    def calculateColor(self, obj, point, eye):
        pointNormal = obj.normal(point)
        result = ambientColor

        for light in lights:
            # Light direction
            lightDir = normalize(light.position - point)

            # Check if point is blocked
            tmin = float("inf")
            for obj in objects:
                t1, t2 = obj.intersect(point, lightDir)
                if t1 > 0 and t1 < tmin:
                    break
            else:
                # Lambertian
                ld = light.intensity * np.maximum(0.0, np.dot(pointNormal,lightDir))
                result = result + ld
        result = self.diffuse_color * result

        # Phong
        if self.exponent is not None:
            eyeDir = normalize( eye - point )
            halfway = normalize( eyeDir+lightDir )
            spec = self.specular_color * light.intensity * np.maximum(0.0, np.dot(pointNormal,halfway)) ** self.exponent 
            result = result + spec
    
        return Color( result )       

class Object:
    def __init__(self, shader):
      self.shader = shader
    
    def intersect(self, rO, rD):
        pass

class Sphere(Object):
    def __init__(self, shader, center, radius):
        super().__init__(shader)
        self.center = center
        self.radius = radius
    
    def intersect(self, rO, rD):
        t = np.dot((self.center-rO),rD)
        point = rO + rD * t
        l = np.linalg.norm(self.center-point)
        deltat = np.sqrt(self.radius**2 - l**2)
        t1 = t-deltat
        t2 = t+deltat
        return t1, t2

    def normal(self, point):
        return np.array([
            (point[0]-self.center[0])/self.radius,
            (point[1]-self.center[1])/self.radius,
            (point[2]-self.center[2])/self.radius ])

class Box(Object):
    def __init__(self, shader, minPt, maxPt):
        super().__init__(shader)
        self.minPt = minPt
        self.maxPt = maxPt
    
    # Doesn't work
    def intersect(self, rO, rD):
        tnear = -float("inf") 
        tfar = float("inf")
        t1 = np.array([0,0,0])
        t2 = np.array([0,0,0])

        for i in range(3):
            if rD[i] == 0:
                if rD[i] < self.minPt[i] or rD[i] > self.maxPt[i]:
                    return -1,-1 
            else: 
                t1[i] = (self.minPt[i] - rO[i]) / rD[i]
                t2[i] = (self.maxPt[i] - rO[i]) / rD[i]

                if t1[i] > t2[i]:
                    t1,t2 = t2,t1
                if t1[i] > tnear:
                    tnear = t1[i]
                if t2[i] < tfar:
                    tfar = t2[i]
                if tnear > tfar or tfar < 0:
                    return -1,-1
        return tnear, tfar
    
    # Doesn't work
    def normal(self, point):
        c = (self.minPt + self.maxPt) * 0.5
        p = point - c
        d = (self.minPt - self.maxPt) * 0.5
        bias = 9.999999994

        return normalize( 
            np.array([
            (p[0] / abs(d[0]) * bias),
            (p[1] / abs(d[1]) * bias),
            (p[2] / abs(d[2]) * bias)]))

class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

# Image info
width = 300
height = 300
img = None

# Scene info
shaders = {}
objects = []
lights  = []

def importXML():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    global viewPoint, viewDir, viewProjNormal, projDistance, viewWidth, viewHeight
    for c in root.findall('camera'):
        viewPoint=np.array(c.findtext('viewPoint').split()).astype(np.float)
        viewDir=np.array(c.findtext('viewDir').split()).astype(np.float)
        viewProjNormal=np.array(c.findtext('projNormal').split()).astype(np.float)
        projDistance=float(c.findtext('projDistance') or 1.0)
        viewWidth=float(c.findtext('viewWidth'))
        viewHeight=float(c.findtext('viewHeight'))
        print('Camera')
        print("\t" + 'viewpoint', viewPoint)
        print("\t" + 'viewDir', viewDir)
        print("\t" + 'viewProjNormal', viewProjNormal)
        print("\t" + 'projDistance', projDistance)
        print("\t" + 'viewWidth', viewWidth)
        print("\t" + 'viewHeight', viewHeight)

    global objects, shaders, lights
    for c in root.findall('shader'):
        diffuse_color=np.array(c.findtext('diffuseColor').split()).astype(np.float)
        name = c.get('name')
        type = c.get('type')
        shader = Shader(diffuse_color)
        if type == "Phong":
            shader.specular_color = np.array(c.findtext('specularColor').split()).astype(np.float)
            shader.exponent = int(c.findtext('exponent'))
        shaders[name] = shader
        print('Shader: ', name)
        print("\t" + 'diff', shader.diffuse_color)
        print("\t" + 'spec', shader.specular_color)
        print("\t" + 'expo', shader.exponent)

    for c in root.findall('surface'):
        ref = c.find('shader').get('ref')
        shader = shaders[ref]
        type = c.get('type')
        print( 'type', str(type) )
        if type == "Sphere":
            center = np.array(c.findtext('center').split()).astype(np.float)
            radius = int(c.findtext('radius'))
            objects.append( Sphere(shader, center, radius) )
            print( "\t" + 'center', str(center) )
            print( "\t" + 'radius', str(radius) )
            print( "\t" + 'shader', str(ref) )
        elif type == "Box":
            minPt = np.array(c.findtext('minPt').split()).astype(np.float)
            maxPt = np.array(c.findtext('maxPt').split()).astype(np.float)
            objects.append( Box(shader, minPt, maxPt) )
            print( "\t" + 'minPt', str(minPt) )
            print( "\t" + 'maxPt', str(maxPt) )
            print( "\t" + 'shader', str(ref) )
        
    for c in root.findall('light'):
        position  = np.array(c.findtext('position' ).split()).astype(np.float)
        intensity = np.array(c.findtext('intensity').split()).astype(np.float)
        lights.append( Light(position, intensity) )
        print( 'Light' )
        print( "\t" + 'position:  ', str(position ) )
        print( "\t" + 'intensity: ', str(intensity) )
 
    global width, height
    imgSize=np.array(root.findtext('image').split()).astype(np.int)
    width  = imgSize[0]
    height = imgSize[1]

def normalize(vec):
    return vec / np.linalg.norm(vec)

def rasterToWorld(x, y, width, height):
    aspect = width/height
    u = (2.0 * ((x + 0.5) / width) - 1.0) * aspect
    v = (1.0 - 2.0 * ((y + 0.5) / height))
    return u * viewWidth/2 , v * viewHeight/2

def setColor(x, y, color):
    color.gammaCorrect(2.2)
    img[y][x] = color.toUINT8()

def main(): 
    # Import data
    importXML()
        
    # Create an empty image
    global img
    channels = 3
    img = np.zeros((height, width, channels), dtype=np.uint8)
    img[:,:]=0

    # Get perspective info
    p = e = viewPoint
    u = normalize( np.cross(viewDir, viewUp) )
    v = normalize( np.cross(u, viewDir) )
    w = normalize( -viewDir )

    # For each pixel
    for x in np.arange(width):
        for y in np.arange(height):
            # Perspective ray info
            i, j = rasterToWorld(x, y, width, height)
            s = e + i*u + j*v - w*projDistance
            d = normalize(s - e)

            # Find closest object intersection
            tmin = float("inf")
            tmp  = None
            for obj in objects:
                t1, t2 = obj.intersect(p, d)
                if t1 > 0 and t1 < tmin:
                    tmin = t1
                    tmp  = obj
            
            # If hit object calculate color by shader
            if tmp:
                point = p+tmin*d 
                col = tmp.shader.calculateColor(tmp, point, e)
                setColor(x, y, col)

    print(width, "x", height)

    rawimg = Image.fromarray(img, 'RGB')
    #rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')
    
if __name__=="__main__":
    main()
 