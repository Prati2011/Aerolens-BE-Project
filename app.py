import json, time, cv2, torch, argparse, flask, logging, socket
from serial import Serial
from flask_caching import Cache
from collections import deque
from ast import literal_eval
from queue import Queue
from ultralytics import YOLO
from cvzone import cornerRect
from threading import Thread, Event, Lock
from numpy import interp, ndarray, sqrt, add, square, subtract, uint
from flask_cors import CORS
from cv2 import (
    VideoCapture,
    destroyAllWindows,
    ellipse,
    line,
    imencode,
    arrowedLine,
)

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--nodemcu", action="store_true")
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--maxdet", default=1, type=int)
parser.add_argument("--conf", default=0.5, type=float)
parser.add_argument("--port", default=8000, type=int)
parser.add_argument("--host", default="0.0.0.0", type=str)
parser.add_argument("--classes", default="none", type=str)
parser.add_argument("--linethickness", default=2, type=int)
parser.add_argument(
    "--source",
    default="F35.mp4",
    type=str,
)
parser.add_argument(
    "--weights",
    type=str,
    default="best.pt",
)
args = parser.parse_args()


SOURCE = args.source if not args.source.isdigit() else int(args.source)
DEVICE = torch.device(
    (int(args.device) if args.device.isdigit() else args.device)
    if torch.cuda.is_available()
    else "cpu"
)
CLASSES = literal_eval(args.classes) if args.classes.lower() != "none" else None
name = "ESP-DC231A.mshome.net"
NODE_MCU_IP = "0.0.0.0"

if args.nodemcu:
    while True:
        try:
            addr = socket.gethostbyname(name)
            # socket.create_connection(addr)
            NODE_MCU_IP = addr
            break
        except:
            print("Error connecting Node Mcu Retrying")
            pass
        time.sleep(0.2)
HOST = args.host
PORT = args.port
MAX_DET = args.maxdet
CONFICENCE = args.conf
WEIGHTS = args.weights
LINE_THICKNESS = args.linethickness
ELLIPSE_START_ANGLE = 100
ELLIPSE_END_ANGLE = 260
RGB_RED_COLOR = 0, 50, 255
RGB_GREEN_COLOR = 69, 245, 5
NEUTRAL_COLOR = 121, 182, 242

exit_flag: Event = Event()
queue_1 = Queue(20)
queue_2 = Queue(20)
queue_3 = Queue(20)
queue_4 = Queue(20)
queue_5 = Queue(20)
queue_6 = Queue()

deep_learning_model = YOLO(model=WEIGHTS)
class_names = json.dumps(deep_learning_model.names)


app = flask.Flask(__name__)
cache = Cache(app)
app.config["CACHE_TYPE"] = "simple"
CORS(app)


@app.route("/")
def index():
    return flask.send_file("templates/index.html")


@app.route("/get-class-names")
def get_class_names():
    return class_names


def distance_between_points(x1: ndarray[float, float], x2: ndarray[float, float]):
    return sqrt(
        add(
            square(subtract(x1[0], x2[0])),
            square(subtract(x1[1], x2[1])),
        )
    )


def draw_inference_results(exit_flag: Event, app: flask.Flask):
    def genrate_frames():
        while not exit_flag.is_set():
            frame, xyxy, xywh = queue_1.get()

            original_frame = frame
            original_frame_height, original_frame_width = original_frame.shape[:2]
            ellipse_xcenter = original_frame_width // 8
            ellipse_ycenter = original_frame_height // 2
            ellipse_major_length = original_frame_width // 12
            ellipse_minor_length = original_frame_height // 3
            original_frame_center = (
                original_frame_width // 2,
                original_frame_height // 2,
            )

            ellipse(
                original_frame,
                (ellipse_xcenter, ellipse_ycenter),
                (ellipse_major_length, ellipse_minor_length),
                0,
                ELLIPSE_START_ANGLE,
                ELLIPSE_END_ANGLE,
                NEUTRAL_COLOR,
                LINE_THICKNESS,
            )

            ellipse(
                original_frame,
                (original_frame_width - ellipse_xcenter, ellipse_ycenter),
                (ellipse_major_length, ellipse_minor_length),
                180,
                ELLIPSE_START_ANGLE,
                ELLIPSE_END_ANGLE,
                NEUTRAL_COLOR,
                LINE_THICKNESS,
            )
            line(
                original_frame,
                (original_frame_width // 2, (original_frame_height // 2) - 24),
                (original_frame_width // 2, (original_frame_height // 2) + 24),
                NEUTRAL_COLOR,
                LINE_THICKNESS,
            )
            line(
                original_frame,
                ((original_frame_width // 2) - 24, original_frame_height // 2),
                ((original_frame_width // 2) + 24, original_frame_height // 2),
                NEUTRAL_COLOR,
                LINE_THICKNESS,
            )

            for xyxy_array, xywh_array in zip(xyxy, xywh):
                for xywh, xyxy in zip(
                    xywh_array.astype(uint),
                    xyxy_array.astype(uint),
                ):
                    if distance_between_points(xywh[:2], original_frame_center) < 80:
                        COLOR = RGB_GREEN_COLOR
                    else:
                        COLOR = RGB_RED_COLOR
                        arrowedLine(
                            original_frame,
                            original_frame_center,
                            (xywh[0], xywh[1]),
                            COLOR,
                            LINE_THICKNESS + 2,
                        )

                    ellipse(
                        original_frame,
                        (ellipse_xcenter, ellipse_ycenter),
                        (ellipse_major_length, ellipse_minor_length),
                        0,
                        ELLIPSE_START_ANGLE,
                        ELLIPSE_END_ANGLE,
                        COLOR,
                        LINE_THICKNESS,
                    )

                    ellipse(
                        original_frame,
                        (original_frame_width - ellipse_xcenter, ellipse_ycenter),
                        (ellipse_major_length, ellipse_minor_length),
                        180,
                        ELLIPSE_START_ANGLE,
                        ELLIPSE_END_ANGLE,
                        COLOR,
                        LINE_THICKNESS,
                    )
                    line(
                        original_frame,
                        (original_frame_width // 2, (original_frame_height // 2) - 24),
                        (original_frame_width // 2, (original_frame_height // 2) + 24),
                        COLOR,
                        LINE_THICKNESS + 2,
                    )
                    line(
                        original_frame,
                        ((original_frame_width // 2) - 24, original_frame_height // 2),
                        ((original_frame_width // 2) + 24, original_frame_height // 2),
                        COLOR,
                        LINE_THICKNESS + 2,
                    )

                    cornerRect(
                        original_frame,
                        (xyxy[0], xyxy[1], xywh[2], xywh[3]),
                        colorC=COLOR,
                    )

                ret, jpeg = imencode(".jpg", original_frame)

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )

    @app.route("/image-result-feed")
    def vid_feed():
        return flask.Response(
            genrate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame" 
        )


def distance_chart_server(exit_flag: Event, app: flask.Flask):
    def genrate_data():
        while not exit_flag.is_set():
            (
                xywh_item,
                xyxy_item,
                frame_width,
                frame_height,
            ) = queue_3.get()

            for xywh_array, xyxy_array in zip(xywh_item, xyxy_item):
                for xywh, xyxy in zip(xywh_array, xyxy_array):
                    yield (
                        f'data: {{"x":{xywh[0]},"y":{xywh[1]},"w":{xywh[2]},'
                        f'"frameWidth":{frame_width},"frameHeight":{frame_height}}}\n\n'
                    )

    @app.route("/distance-chart")
    def distance_chart_send():
        return flask.Response(genrate_data(), content_type="text/event-stream")


def speed_cls_conf_server(exit_flag, app):
    def genrate_data():
        while not exit_flag.is_set():
            speed, cls, conf = queue_4.get()

            yield (
                f'data: {{"speed":{speed},"cls":{cls},'
                f'"conf":{conf}}}\n\n'.replace("'", '"')
            )

    @app.route("/get-speed-cls-conf")
    def send_speed_cls_conf():
        return flask.Response(genrate_data(), content_type="text/event-stream")


def send_original_frame(exit_flag, app):
    def genrate_frames():
        while not exit_flag.is_set():
            frame = queue_5.get()

            ret, jpeg = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

    @app.route("/get-original-feed")
    def send_original_feed():
        return flask.Response(
            genrate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )


# node_mcu_dqueue = deque(maxlen=3)
node_mcu_dqueue = Queue(maxsize=2)
# node_mcu_dqueue = []
node_mcu_lock = Lock()


def node_mcu_position(exit_flag: Event):
    import requests, time

    nodeMCU_ip = NODE_MCU_IP
    # nodeMCU_ip = '192.168.137.221'
    url = f"http://{nodeMCU_ip}/setservo"
    urlx = f"http://{nodeMCU_ip}/setx"
    urly = f"http://{nodeMCU_ip}/sety"

    def set_position(x, y):
        payload = {"pan": str(x), "tilt": str(y)}
        response = requests.post(url, data=payload)
        # Print the response from NodeMCU
        # print(response.text)

    def setX(x):
        payload = {"pan": str(x)}
        response = requests.post(urlx, data=payload)

    def setY(y):
        payload = {"tilt": str(y)}
        response = requests.post(urly, data=payload)

    while not exit_flag.is_set():
        # time.sleep(0.015)
        with node_mcu_lock:
            # if not len(node_mcu_dqueue):
            #     continue
            # xywh_array, fw, fh = node_mcu_dqueue.popleft()
            time.sleep(0.03)
            xywh_array, fw, fh = node_mcu_dqueue.get()
            # if len(node_mcu_dqueue) >= 3:
            # node_mcu_dqueue = []
            # if len(node_mcu_dqueue) < 1: continue
            # xywh_array, fw, fh = node_mcu_dqueue[-1]
            fw_center, fh_center = fw // 2, fh // 2
            prevx, prey = 0, 0
            for xywh_values in xywh_array:
                for xywh in xywh_values:
                    # time.sleep(2)
                    x = interp(fw - xywh[0], (0, fw), (0, 180))
                    y = interp(fh - xywh[1], (0, fh), (100, 190))
                    set_position(int(x), int(y))


def db_server(app):
    from flask import Flask, jsonify, request
    from pymongo import MongoClient

    db = MongoClient("mongodb://localhost:27017")["aircraft"]
    collection = db["aircraft_length"]

    @app.route("/get-aircraft-length")
    @cache.cached(timeout=30)
    def send_name():
        name = request.args.get("name").upper()
        result = collection.find_one({"name": name}, ["length"])
        if result:
            return jsonify({"length": result["length"]})
        else:
            return jsonify({"length": None})

    @app.route("/get-focal-length")
    @cache.cached(timeout=30)
    def send_focal_length():
        return jsonify({"f": 550})


db_server_thread = Thread(
    target=db_server,
    daemon=True,
    args=(app,),
)
node_mcu_position_thread = Thread(
    target=node_mcu_position,
    daemon=True,
    args=(exit_flag,),
)
draw_inference_results_thread = Thread(
    target=draw_inference_results,
    daemon=True,
    args=(
        exit_flag,
        app,
    ),
)
distance_chart_server_thread = Thread(
    target=distance_chart_server,
    daemon=True,
    args=(
        exit_flag,
        app,
    ),
)
speed_cls_conf_server_thread = Thread(
    target=speed_cls_conf_server,
    daemon=True,
    args=(
        exit_flag,
        app,
    ),
)
send_original_frame_thread = Thread(
    target=send_original_frame,
    args=(
        exit_flag,
        app,
    ),
)
flask_app_thread = Thread(
    target=lambda app: app.run(
        debug=False,
        port=PORT,
        host=HOST,
        use_reloader=False,
    ),
    daemon=True,
    args=(app,),
)

# db_server_thread.start()
time.sleep(0.2)
if args.nodemcu:
    node_mcu_position_thread.start()
time.sleep(0.2)
distance_chart_server_thread.start()
time.sleep(0.2)
speed_cls_conf_server_thread.start()
time.sleep(0.2)
send_original_frame_thread.start()
time.sleep(0.2)
draw_inference_results_thread.start()
time.sleep(0.2)
flask_app_thread.start()
time.sleep(0.2)

video_capture = VideoCapture(SOURCE)
time.sleep(3)

try:
    while video_capture.isOpened() and not exit_flag.is_set():
        success, frame = video_capture.read()
        if not success:
            continue

        inference_results = deep_learning_model.track(
            frame,
            max_det=MAX_DET,
            classes=CLASSES,
            conf=CONFICENCE,
            device=DEVICE,
            verbose=False,
        )

        speed = list(result.speed for result in inference_results)
        cls = list(
            result.boxes.cls.cpu().numpy().tolist() for result in inference_results
        )
        conf = list(
            result.boxes.conf.cpu().numpy().tolist() for result in inference_results
        )
        xywh = list(result.boxes.xywh.cpu().numpy() for result in inference_results)
        xyxy = list(result.boxes.xyxy.cpu().numpy() for result in inference_results)

        queue_5.put(frame.copy())
        queue_1.put((frame, xyxy, xywh))
        # with node_mcu_lock:
        # node_mcu_dqueue.append((xywh, frame.shape[1], frame.shape[0]))
        if node_mcu_dqueue.empty():
            node_mcu_dqueue.put((xywh, frame.shape[1], frame.shape[0]))
        queue_3.put((xywh, xyxy, frame.shape[1], frame.shape[0]))
        queue_4.put((speed, cls, conf))
        time.sleep(0.01)

except KeyboardInterrupt:
    exit_flag.set()
    print("Exit Flag Set")
    video_capture.release()
    destroyAllWindows()
    print("Existed Successfully")
    exit()
