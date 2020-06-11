#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:05:46 2019

@author: chengxue
"""

from computer_vision.ImageSegmenter import ImageSegmenter
import cv2
import numpy as np
import time
from computer_vision.cv_utils import Rectangle
from computer_vision.game_object import GameObject, GameObjectType
import sys
sys.path.append('..')


class VisionRealShape:
    
    def __init__(self,screenshot):
        #self.screenshot = screenshot[:,:,::-1]
        self.ImageSegmenter = ImageSegmenter(screenshot)
        self.ImageSegmenter._findEdges()
        self.connectComp = self.ImageSegmenter.findConnectedComponents()
        self.allObj = self.findAllObj()

    def findAllObj(self):
        '''
        let's only return the bounding box for the agents agent
        '''
        allObj = {}
        object_type =  {2:'hill',3:'slingshot',4:'redBird',5:'yellowBird',6:'blueBird',7:'blackBird',8:'whiteBird',
                        9:'pig',10:'ice',11:'wood',12:'stone'}
        for c in range(2,13):
            to_ret = np.zeros((self.ImageSegmenter._height,self.ImageSegmenter._width)).astype(np.uint8)
            to_ret[self.connectComp==c] = 255
            contours, hierarchy = cv2.findContours(to_ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            res_indi = {c:{object_type[c]:[]}}
            for con in contours:
                if cv2.contourArea(con) >= self.ImageSegmenter.MIN_SIZE[c] and cv2.contourArea(con) <= self.ImageSegmenter.MAX_SIZE[c]:
                    x,y,w,h = cv2.boundingRect(con)
                    if x >= 50 and y >=100:
                        #print(x,y,c)
                        if c == 3 and x < 240 and y > 240:
                            #print(x,y,c)
                            box = Rectangle((np.array([y+h,y]),np.array([x+w,x])))
                            
                            box.width,box.height = box.height,box.width
                            

                            #check the aspect ration
                            if box.height/ box.width>2:
                            
                                game_object = GameObject(box, GameObjectType(object_type[c]))
                                res_indi[c][object_type[c]].append(game_object)
                        elif c == 9:
                            #pigs can not overlap
                            box = Rectangle((np.array([y+h,y]),np.array([x+w,x])))
                            box.width,box.height = box.height,box.width
                            game_object = GameObject(box, GameObjectType(object_type[c]))
                            res_indi[c][object_type[c]].append(game_object)

                        elif c!= 3:
                            box = Rectangle((np.array([y+h,y]),np.array([x+w,x])))
                            box.width,box.height = box.height,box.width
                            game_object = GameObject(box, GameObjectType(object_type[c]))
                            res_indi[c][object_type[c]].append(game_object)
                            

            allObj.update(res_indi)
        return allObj
            
    def find_bird_on_sling(self,birds,sling):
        sling_top_left = sling.top_left[1]
        distance = {}
        for bird_type in birds:
            if len(birds[bird_type]) > 0:
                for bird in birds[bird_type]:
                    #print(bird)
                    distance[bird] = abs(bird.top_left[1]\
                                    - sling_top_left)

        min_distance = 1000
        for bird in distance:
            if distance[bird] < min_distance:
                ret = bird
                min_distance = distance[bird]

        return ret
    
    def find_hill_mbr(self):
        ret = self.allObj[2]['hill']
        #ret = self.remove_overlap(ret,5)
        return ret        
        
            
    def find_pigs_mbr(self):
        ret = self.allObj[9]
        ret = self.remove_overlap(ret,10)
        return ret['pig']
    
    def find_slingshot_mbr(self):
        return self.allObj[3]['slingshot']

    def find_birds(self):
        ret = {}
        for i in range(4,9):
            ret.update(self.allObj[i])
        
        #remove over lapping birds
        ret = self.remove_overlap(ret,10)
        
        return ret

    def find_blocks(self):
        ret = {}
        for i in range(10,13):
            ret.update(self.allObj[i])
        return ret
    
    def remove_overlap(self,ret,distance):
        
        objects = []
        final = {}
        for otype in ret:
            for obj in ret[otype]:
                objects.append([obj,otype])
            final[otype] = []
        
        if len(objects) == 1:
            return ret
        
        ignore = []
        for i in range(len(objects)):
            if i not in ignore:
                for j in range(i,len(objects)):
                    if j not in ignore:
                        #print(i,j)
                        if abs(objects[i][0].get_centre_point()[0] - objects[j][0].get_centre_point()[0]) <= distance and \
                        abs(objects[i][0].get_centre_point()[1] - objects[j][0].get_centre_point()[1]) <= distance:
                            #print(i,j)
                            if objects[i][0] not in final[objects[i][1]]:
                                final[objects[i][1]].append(objects[i][0])
                            ignore.append(j)
                        else:
                            if objects[i][0] not in final[objects[i][1]]:
                                final[objects[i][1]].append(objects[i][0])        
        return final
                                
            
if __name__ == "__main__":
    
    img = cv2.imread('../demo/'+'screenshot.png')
    t = time.time()
    img = img[:,:,::-1]
    imageSeg = VisionRealShape(img)
    img = img[:,:,::-1]
    birds = imageSeg.find_birds()
    sling = imageSeg.find_slingshot_mbr()
    pigs = imageSeg.find_pigs_mbr()
    blocks = imageSeg.find_blocks()
    hill = imageSeg.find_hill_mbr()
    objects = [sling,birds,pigs,blocks,hill]
    objects_dict = {}
    for obj in objects:
        objects_dict.update(obj)
    for object_type in objects_dict:
        for obj in objects_dict[object_type]:
            cv2.rectangle(img, tuple(obj.top_left),tuple(obj.bottom_right),(255,0,0),1)
            cv2.putText(img,object_type,tuple(obj.top_left),0,0.3,(0,0,255))
    
    cv2.imshow('res',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
