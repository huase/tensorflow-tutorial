#!/usr/bin/env python

# export GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>

import argparse
import base64

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

def main(confidence,maxResults):
    for d in ['data/images/train','data/images/val','data/images/test']:
        getObjects(d,confidence,maxResults)

def getObjects(d,confidence,maxResults):
    import os
    import cPickle as pickle

    print "STARTING..."

    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    group = d.split('/')[-1]

    if group == 'train':
        labelMap = {}
        pictures = {}

        # i = 0
        for root, dirs, files in os.walk(d):
            for file in files[:10]:
                if file.endswith(".jpg"):
                    # i+=1
                    picture = os.path.join(root,file)
                    print 'PROCESSING: ',picture 
                    try:
                        labels = annotate(picture,confidence,maxResults,service)
                    except:
                        continue
                    for label in labels:
                        if label not in labelMap:
                            labelMap[label] = len(labelMap)
                    pictures[picture] = labels
            # if i > 10:
            #     break

        pickle.dump(labelMap, open("labelMap.p","wb"))
        pickle.dump(pictures, open("%s-labels.p"%group,"wb")) 

        # print labelMap

    else:
        pictures = {}

        # i = 0
        for root, dirs, files in os.walk(d):
            for file in files[:10]:
                if file.endswith(".jpg"):
                    # i+=1
                    picture = os.path.join(root,file)
                    print 'PROCESSING: ',picture 
                    try:
                        labels = annotate(picture,confidence,maxResults,service)
                    except:
                        continue
                    pictures[picture] = labels
            # if i > 10:
            #     break

        pickle.dump(pictures, open("%s-labels.p"%group,"wb")) 


def annotate(photo_file,confidence,maxResults,service):
    labels = []
    with open(photo_file, 'rb') as image:
        image_content = base64.b64encode(image.read())
        service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image_content.decode('UTF-8')
                },
                'features': [{
                    'type': 'LABEL_DETECTION',
                    'maxResults': maxResults
                }]
            }]
        })
        response = service_request.execute()
        if 'labelAnnotations' in response['responses'][0]:
            for label in response['responses'][0]['labelAnnotations']:
                if label['score'] >= confidence:
                    labels.append(str(label['description']))
    return labels

def recover(p):
    import cPickle as pickle

    d = pickle.load(open(p, "rb"))
    print d


if __name__ == '__main__':
    main(0.65,20)
    # recover('labelMap.p')


