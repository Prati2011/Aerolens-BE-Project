from torch import nn
from queue import Queue
from torchvision import transforms
from threading import Event, Thread
from ast import literal_eval
import flask, flask_cors, cv2, torch, numpy as np, argparse

# making parser
parser = argparse.ArgumentParser(
    description="A Flask based Program to show the output of conv2d"
)

parser.add_argument(
    "--source",
    type=str,
    default="C:/Users/vishal/DriveD/BE-Project/test_images/F35.mp4",
    help="source to read frame else defaults to webcam",
)

args = parser.parse_args()
source = int(args.source) if args.source.isdigit() else args.source

DEVICE = torch.device("cpu")

exit_event = Event()
app = flask.Flask(__name__)
flask_cors.CORS(app)

frame_queue = Queue(20)
original_queue = Queue(20)
conv_one_queue = Queue(20)
conv_two_queue = Queue(20)

conv_layer_1 = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    nn.BatchNorm2d(3, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
    nn.SiLU(inplace=False),
).to(device=DEVICE)

conv_layer_2 = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    nn.BatchNorm2d(3, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
    nn.SiLU(inplace=False),
).to(device=DEVICE)

conv_layer_3 = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
    nn.BatchNorm2d(3, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
    nn.SiLU(inplace=False),
).to(device=DEVICE)

transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
).to(device=DEVICE)

to_tensor = transforms.ToTensor()


@app.route("/")
def index():
    return flask.send_file("templates/convo.html")


def genrate_orig_frames():
    while not exit_event.is_set():
        frame = original_queue.get()
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


@app.route("/orig-img")
def orig_img():
    return flask.Response(
        genrate_orig_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def genrate_frames_of_conv_1():
    while not exit_event.is_set():
        frame = frame_queue.get()
        frame = transform(to_tensor(frame).to(device=DEVICE)).unsqueeze(0)
        output_conv_1 = conv_layer_1(frame)
        conv_one_queue.put(output_conv_1)
        for data in output_conv_1.data.cpu().numpy():
            output_np = data.transpose((1, 2, 0))
            output_np = cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB)
            output_uint8 = (output_np * 255).astype(np.uint8)
            ret, jpeg = cv2.imencode(".jpg", output_uint8)
            if not ret:
                continue
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


@app.route("/get-conv-output-1")
def convo_output_1_handler():
    return flask.Response(
        genrate_frames_of_conv_1(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def genrate_frames_of_conv_2():
    while not exit_event.is_set():
        output_conv_1 = conv_one_queue.get()
        output_conv_2 = conv_layer_2(output_conv_1)
        conv_two_queue.put(output_conv_2)
        for data in output_conv_2.data.cpu().numpy():
            output_np = data.transpose((1, 2, 0))
            output_np = cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB)
            output_uint8 = (output_np * 255).astype(np.uint8)
            ret, jpeg = cv2.imencode(".jpg", output_uint8)
            if not ret:
                continue
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


@app.route("/get-conv-output-2")
def convo_output_2_handler():
    return flask.Response(
        genrate_frames_of_conv_2(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def genrate_frames_of_conv_3():
    while not exit_event.is_set():
        output_conv_2 = conv_two_queue.get()
        output_conv_3 = conv_layer_3(output_conv_2)
        for data in output_conv_3.data.cpu().numpy():
            output_np = data.transpose((1, 2, 0))
            output_np = cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB)
            output_uint8 = (output_np * 255).astype(np.uint8)
            ret, jpeg = cv2.imencode(".jpg", output_uint8)
            if not ret:
                continue
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


@app.route("/get-conv-output-3")
def convo_output_3_handler():
    return flask.Response(
        genrate_frames_of_conv_3(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


flask_server_thread = Thread(
    target=lambda: app.run(
        host="0.0.0.0",
        port=8080,
        debug=False,
    ),
    daemon=True,
)

flask_server_thread.start()

# cap = cv2.VideoCapture("C:\\Users\\vishal\\DriveD\\BE-Project\\test_images\\F35.mp4")
cap = cv2.VideoCapture(source)

while not exit_event.is_set():
    ret, frame = cap.read()
    if not ret:
        continue
    original_queue.put(frame.copy())
    frame_queue.put(frame)

cap.release()
cv2.destroyAllWindows()
