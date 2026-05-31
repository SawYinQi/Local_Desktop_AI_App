import grpc
import agent_pb2
import agent_pb2_grpc

'''
temporary client for testing 
'''
def run():
    # open connection to local server port
    channel = grpc.insecure_channel("localhost:50051")

    # translate function calls to network request
    stub = agent_pb2_grpc.AgentServiceStub(channel)

    upload = stub.UploadVideo(
        agent_pb2.VideoRequest(session_id="test-1", file_path="sample.mp4")
    )

    print("Upload: ", upload.message) # log server response to upload

    # send query to server
    responses = stub.SendQuery(
        agent_pb2.QueryRequest(session_id="test-1", query="what was by previous question about?")
    )

    # Handles streaming responses 
    print("Server replied: ")
    for response in responses:
        print(response.response)


if __name__ == "__main__":
    run()