from concurrent import futures
import grpc

import agent_pb2
import agent_pb2_grpc

'''
Class AgentService inherits agent_pb2_grpc.AgentServiceServicer
Overrides placeholder in parent class with Upload/Query Logic
'''
class AgentService(agent_pb2_grpc.AgentServiceServicer):

    def UploadVideo(self, request, context):
        # log session id + file_path
        print(f"UploadVideo: session={request.session_id} path={request.file_path}")

        # return response
        return agent_pb2.VideoResponse(
            # dummy values
            success=True, 
            message=f"Received video at {request.file_path}", # status msg
        )

    
    def SendQuery(self, request, context):
        # log session id + query 
        print(f"SendQuery: session={request.session_id} query={request.query}")
        # return yeild response stream
        yield agent_pb2.QueryResponse(
            #dummy values
            response=f"You asked: '{request.query}'. (Backend is alive!)", 
            needs_clarification=False, 
            clarification_prompt="",
            artifact_path="",
        )


def serve():
    # create server 
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # add overwritten AgentService logic to gRPC server engine
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentService(), server)
    # server listen on port 50051
    server.add_insecure_port("[::]:50051")
    # start server
    server.start()
    print("gRPC server running on port 50051.")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()