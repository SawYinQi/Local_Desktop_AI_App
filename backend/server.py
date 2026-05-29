from concurrent import futures
import grpc

import agent_pb2
import agent_pb2_grpc
from mcp_client import call_tool

# temporary dictionary memory of video session id : video_path
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
            # dummy values
            success=True, 
            message=f"Received video at {request.file_path}", # status msg
        )

    
    def SendQuery(self, request, context):
        query =  request.query.lower()
        session = request.session_id
        # log session id + query 
        print(f"SendQuery: session={session} query={query}")

        video_path = session_videos.get(session) # get video path for session

        # simple keyword-based routing of query to agent function; replace with LLM orchestration later on 
        if "transcribe" in query or "transcript" in query:

            # if no video uploaded for session, return error response
            if not video_path:
                yield agent_pb2.QueryResponse(response="No video uploaded yet. Please upload a video first.")
                return
            
            # if video exist, call mcp client to run transcription tool and stream response back to client; replace with LLM orchestration later on
            yield agent_pb2.QueryResponse(response="Transcribing...")
            transcript = call_tool("transcription", "transcribe_video", {"file_path": video_path})
            yield agent_pb2.QueryResponse(response=transcript)
            return
        
        # fallback for unknown queries
        yield agent_pb2.QueryResponse(response=f"I received: '{request.query}', but I can't handle that yet.")

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