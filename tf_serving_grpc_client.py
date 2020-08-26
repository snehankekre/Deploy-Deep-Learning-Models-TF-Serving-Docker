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
    """ no error handling at all, just poc"""
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
