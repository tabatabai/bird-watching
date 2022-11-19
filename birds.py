import streamlink
import torch
import torch.nn as nn
from twitchAPI.twitch import Twitch
from multiprocessing import Queue, Process
import cv2
import time
import datetime
import json
from telegram.ext import Updater


class TwitchInfo:
    def __init__(self, streamer: str) -> None:
        """Credentials of Twitch App"""
        with open("app_id.txt", "r") as f:
            app_id = f.read().strip()
        with open("app_secret.txt", "r") as f:
            app_secret = f.read().strip()
        self.twitch = Twitch(app_id, app_secret)
        self.streamer = streamer
        self.user_id = self.twitch.get_users(logins=self.streamer)["data"][0]["id"]

    def get_current_video_url(self):
        """Kind of hacky way to generate a timestamp based
        on the difference to the start time of the most recent VOD."""
        videos = self.twitch.get_videos(ids=None, user_id=self.user_id)["data"]
        current_video = videos[0]
        created_at = current_video["created_at"]
        created_at = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        now = datetime.datetime.utcnow()
        diff = now - created_at
        h, m, s = self.hours_minutes_seconds(diff)
        return (
            current_video["url"][12:] + f"?t={str(h).zfill(2)}h{str(m).zfill(2)}m{str(s).zfill(2)}s"
        )

    def get_current_stream_id(self):
        streams = self.twitch.get_streams(user_id=self.user_id)["data"]
        if len(streams) == 0:
            return None
        else:
            return streams[0]["id"]

    def hours_minutes_seconds(self, td):
        return (td.seconds // 3600) % 60, (td.seconds // 60) % 60, td.seconds % 60


class FrameGetter:
    def __init__(self, streamer, quality):
        self.streamer = streamer
        self.quality = quality
        self.command_q = Queue()
        self.result_q = Queue()
        capturing_process = Process(target=self._capturing)
        capturing_process.start()

    def get_frame(self):
        self.command_q.put("get_frame")
        return self.result_q.get()

    def end(self):
        self.command_q.put("end")

    def _capturing(self):
        """This process captures all frames, and keeps the most recent
        one in most_recent_frame (this way the buffer in cv2.VideoCapture is emptied).
        This is needed, so that upon request (by sending the command 'get_frame'),
        the most recent frame can be output immediately."""
        streams = streamlink.streams(f"https://www.twitch.tv/{self.streamer}")
        stream_url = streams[self.quality].to_url()
        capture = cv2.VideoCapture(stream_url)
        most_recent_frame = None
        while True:
            try:
                success, frame = capture.read()
            except Exception as e:
                print("Caught unexpected exception: ", e)
                success = False
            if success:
                most_recent_frame = frame
            if not self.command_q.empty():
                cmd = self.command_q.get_nowait()
                if cmd == "get_frame":
                    self.result_q.put(most_recent_frame)
                elif cmd == "end":
                    break


class BirdBot:
    def __init__(self) -> None:
        self.chat_id = -1  # Your Telegram chat id
        with open("token.txt", "r") as token_file:
            """Telegram Bot Token"""
            token_string = token_file.read().strip()
        self.bot = Updater(token_string).bot

    def send_photo(self, photo, caption):
        self.bot.send_photo(self.chat_id, photo, caption)


class Inferencer:
    def __init__(self) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.fix_model(self.model)
        self.last_detection_time = time.time()

    def fix_model(self, model):
        """
        Fix issue described here
        https://github.com/ultralytics/yolov5/issues/6948
        """
        for m in model.modules():
            if isinstance(m, nn.Upsample):
                m.recompute_scale_factor = None

    def check_for_bird(self, frame):
        results = self.model(frame)
        results_json_string = results.pandas().xyxy[0].to_json(orient="records")
        results_json = json.loads(results_json_string)
        for detection in results_json:
            if detection["name"] == "bird":
                delta = time.time() - self.last_detection_time
                self.last_detection_time = time.time()
                return True, delta
        return False, None


if __name__ == "__main__":
    streamer = "Your twitch name"
    twitch_info = TwitchInfo(streamer=streamer)
    frame_getter = None
    bird_bot = BirdBot()
    inferencer = Inferencer()

    current_stream_id = None
    c = 0
    while True:
        time.sleep(1)
        if c == 0:
            new_stream_id = twitch_info.get_current_stream_id()
            print(new_stream_id)
            if new_stream_id != current_stream_id:
                if frame_getter is not None:
                    frame_getter.end()
                    frame_getter = None
                current_stream_id = new_stream_id
                if current_stream_id != None:
                    frame_getter = FrameGetter(streamer=streamer, quality="best")

        if frame_getter is not None:
            frame = frame_getter.get_frame()
            print(type(frame))
            bird_found, detection_delta = inferencer.check_for_bird(frame)
            print("Detection delta:", detection_delta)
            if bird_found and detection_delta > 30:
                video_url = twitch_info.get_current_video_url()
                cv2.imwrite("tmp.jpg", frame)
                caption = (
                    f"I think I found a bird in {streamer}'s current stream!"
                    f"Live: twitch.tv/{streamer} VOD: {video_url}."
                )
                bird_bot.send_photo(open("tmp.jpg", "rb"), caption=caption)

        c = (c + 1) % 60
