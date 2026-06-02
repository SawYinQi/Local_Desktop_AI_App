from concurrent import futures
import grpc

import agent_pb2
import agent_pb2_grpc
import orchestrator

# In-memory store for session video paths. Key: session_id, Value: file_path
session_videos = {}

"""
Class AgentService inherits agent_pb2_grpc.AgentServiceServicer
Overrides placeholder in parent class with Upload/Query Logic
"""
class AgentService(agent_pb2_grpc.AgentServiceServicer):

    def UploadVideo(self, request, context):
        # log session id + file_path
        print(f"UploadVideo: session={request.session_id} path={request.file_path}")
        session_videos[request.session_id] = request.file_path # store video path for session
        # return response
        return agent_pb2.VideoResponse(
            success=True, 
            message=f"Received video at {request.file_path}", # status msg
        )

    
    def SendQuery(self, request, context):
        session = request.session_id
        query = request.query
        print(f"SendQuery: session={session} query={query!r}")

        video_path = session_videos.get(session)
        
        # Delegate handling to the orchestrator
        try:
            for event in orchestrator.handle_query(session_id=session, query=query, video_path=video_path):
                # stream responses back to client as they come in from the orchestrator
                yield agent_pb2.QueryResponse(
                    response=event.get("response", ""),
                    artifact_path=event.get("artifact_path", ""),
                )
        except Exception as e:
            print(f"SendQuery error: {e}")
            yield agent_pb2.QueryResponse(
                response=f"Sorry, something went wrong handling that request: {e}",
                artifact_path="",
            )

def serve():
    # create server 
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # add overridden AgentService logic to gRPC server engine
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentService(), server)
    # server listen on port 50051
    server.add_insecure_port("[::]:50051")
    # start server
    server.start()
    print("gRPC server running on port 50051.")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()