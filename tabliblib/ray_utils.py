import ray
import logging

def start_ray(allow_local=False):
    try:
        # Try to connect to an existing Ray cluster if it's already running.
        ray.init(address='auto')
        print("running ray and connected to the cluster")
    except ConnectionError as ce:
        logging.warning(f"connection error connecting to ray cluster: {ce}")
        if allow_local:
            logging.warning(f"running in local mode; if you encounter errors of the form `task_manager.cc:555:  Check failed: stream_it != object_ref_streams_.end() PeekObjectRefStream API can be used only when the stream has been created and not removed.` this is probably due to local mode. Recommend starting a ray cluster + head node instead.")
            # If no existing Ray cluster, start a new one in local mode.
            ray.init(local_mode=True)
            print("running ray in local mode")
        else:
            raise ce