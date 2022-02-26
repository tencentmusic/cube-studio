
from collections import Counter
import os
import sys
import time
import ray


@ray.remote
def gethostname(x):
    import platform
    import time
    time.sleep(0.01)
    return x + (platform.node(), )


def main():
    # Check that objects can be transferred from each node to each other node.
    for i in range(10):
        print("Iteration {}".format(i))
        results = [
            gethostname.remote(()) for _ in range(100)
        ]
        print(Counter(ray.get(results)))
        sys.stdout.flush()

    print("Success!")
    sys.stdout.flush()


if __name__ == "__main__":
    head_service_ip = os.getenv('RAY_HOST','127.0.0.1')
    head_host = head_service_ip+":10001"
    print(head_host)
    ray.util.connect(head_host)
    main()
