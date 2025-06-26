import redis
import subprocess
 
class RedisHelper:
    def __init__(self):
        self.connection_pool = redis.ConnectionPool(host="127.0.0.1", port=1335)
        self.__conn = redis.Redis(connection_pool=self.connection_pool)
 
    def publish(self, pub, msg):
        self.__conn.publish(pub, msg)
        return True
 
    def subscribe(self, sub):
        pub = self.__conn.pubsub()
        pub.subscribe(sub)
        pub.parse_response()
        return pub

def start_redis(port=1335):
    try:
        check = subprocess.run(['redis-cli', '-p', str(port), 'ping'], capture_output=True, text=True)
        if check.stdout.strip() == 'PONG':
            print(f"Redis is already running on port {port}.")
            return True
        else:
            print(f"Starting Redis on port {port}...")
            subprocess.Popen(['redis-server', '--port', str(port)])
            print("Redis server started.")
            return True
    except Exception as e:
        print(f"Failed to start Redis server: {e}")
        return False
