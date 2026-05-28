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

    # send query to server
    responses = stub.SendQuery(
        agent_pb2.QueryRequest(session_id="test-1", query="transcribe the video")
    )

    # Handles streaming responses 
    for response in responses:
        print("Server replied:", response.response)


if __name__ == "__main__":
    run()