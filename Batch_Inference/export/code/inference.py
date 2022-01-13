from tensorflow.keras.preprocessing import image
import numpy as np
import json
import sys
import requests
from PIL import Image
import io


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    print(sys.getsizeof(data))
    
    if context.request_content_type == 'application/x-image':
        target_size=(224,224)
        img = Image.open(io.BytesIO(data.read()))
        img = img.convert('RGB')
        img = img.resize(target_size, Image.NEAREST)
        img = image.img_to_array(img)
        x = np.expand_dims(img, axis=0)

        return json.dumps({'instances': x.tolist()})
    else:
        _return_error(415, 'Unsupported content type in request "{}"'.format(context.request_content_type or 'Unknown'))


def output_handler(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    print("printing from op handler")
    print(response)
    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = response.content
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))
