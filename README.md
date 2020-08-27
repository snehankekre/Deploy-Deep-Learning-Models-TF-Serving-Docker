# Deploy Deep Learning Models with TensorFlow Serving and Docker
Official repository of [TensorFlow Serving with Docker for Model Deployment](https://www.coursera.org/projects/tensorflow-serving-docker-model-deployment) Coursera Project

### Data Set
The dataset for [TensorFlow Serving with Docker for Model Deployment](https://www.coursera.org/projects/tensorflow-serving-docker-model-deployment) is available [here](https://drive.google.com/file/d/14v42um2VRAfQPfGnBJ4QzOie4VZOPK_F/view?usp=sharing). It is assumed that the contents of the uncompressed file (`train.csv`, `test.csv`) are saved in the sample folder as `train.py` file.


### Getting Started

```
$ virtualenv -p python3 tf-serving-coursera
$ source tf-serving-coursera/bin/activate
$ pip3 install -r requirements.txt
```

### Model Structure

We predict Amazon product ratings based on plaintext reviews.
```python
hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape=[128], 
                 input_shape=[], dtype=tf.string, name='input', trainable=False)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax', name='output'))
model.compile(loss='categorical_crossentropy',
        optimizer='Adam', metrics=['accuracy'])
```


### Test the Model

#### Negative Review

```python
>> test_sentence = "horrible book, waste of time"
>> model.predict([test_sentence])
[0.87390379  0.02980554  0.09629067]
```

#### Positive Review

```python
>> test_sentence = "Awesome product. Loved it! :D"
>> model.predict([test_sentence)
[0.00827967  0.01072392  0.98099641]
```

## Steps to Deploy the Model

#### Export the Model as Protobuf

```python
base_path = "amazon_review/"
path = os.path.join(base_path, str(int(time.time())))
tf.saved_model.save(model, path)
```

### Start TensorFlow Serving with Docker

```
$ docker pull tensorflow/serving

$ docker run -p 8500:8500 \
             -p 8501:8501 \
             --mount type=bind,\
             source=/path/to/amazon_review,\
             target=/models/amazon_review \
             -e MODEL_NAME=amazon_review
             -t tensorflow/serving
```

#### Setup a client (either gRPC or REST based)

```python
import sys
import grpc
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def get_stub(host='127.0.0.1', port='8500'):
    channel = grpc.insecure_channel('127.0.0.1:8500') 
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


def get_model_prediction(model_input, stub, model_name='amazon_review', signature_name='serving_default'):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs['input_input'].CopyFrom(tf.make_tensor_proto(model_input))
    response = stub.Predict.future(request, 5.0)  # 5 seconds
    return response.result().outputs["output"].float_val


def get_model_version(model_name, stub):
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = 'amazon_review'
    request.metadata_field.append("signature_def")
    response = stub.GetModelMetadata(request, 10)
    # signature of loaded model is available here: response.metadata['signature_def']
    return response.model_spec.version.value

if __name__ == '__main__':
    print("\nCreate RPC connection ...")
    stub = get_stub()
    while True:
        print("\nEnter an Amazon review [:q for Quit]")
        if sys.version_info[0] <= 3:
            sentence = raw_input() if sys.version_info[0] < 3 else input()
        if sentence == ':q':
            break
        model_input = [sentence]
        model_prediction = get_model_prediction(model_input, stub)
        print("The model predicted ...")
        print(model_prediction)
```


### Run the client

```
$ python3 tf_serving_grpc_client.py
```

```
Create RPC connection ...

Enter an Amazon review [:q for Quit]
horrible book, waste of time

The model predicted ...
[0.87390379  0.02980554  0.09629067]

Enter an Amazon review [:q for Quit]
Awesome product. Loved it! :D

The model predicted ...
[0.00827967  0.01072392  0.98099641]

Enter an Amazon review [:q for Quit]
:q
```
