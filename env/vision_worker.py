import multiprocessing as mp
import time


class VisionWorker(mp.Process):
    def __init__(self, input_queue, output_queue, ui_detector):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.ui_detector = ui_detector
        self.daemon = True

    def run(self):
        while True:
            frame = self.input_queue.get()

            ui_state = self.ui_detector.detect(frame)

            if not self.output_queue.full():
                self.output_queue.put(ui_state)
