import time


class PerformanceMonitor:

    def __init__(self):
        self.start_times = {}
        self.elapsed_times = {}
        self.display_timer = time.time()

    def start(self, stage):
        self.start_times[stage] = time.perf_counter()

    def stop(self, stage):
        if stage in self.start_times:
            elapsed = (time.perf_counter() - self.start_times[stage]) * 1000
            self.elapsed_times[stage] = elapsed

    def get(self, stage):
        return self.elapsed_times.get(stage, 0.0)

    def total(self):
        return sum(self.elapsed_times.values())

    def reset(self):
        self.elapsed_times.clear()


def display_performance(self):

    current_time = time.time()

    if current_time - self.display_timer >= 1.0:

        print("--------------------------------")
        print(f"Capture          : {self.get('capture'):.2f} ms")
        print(f"FaceMesh         : {self.get('face_mesh'):.2f} ms")
        print(f"EyeMonitor       : {self.get('eye_monitor'):.2f} ms")
        print(f"AttentionMonitor : {self.get('attention_monitor'):.2f} ms")
        print(f"Total            : {self.total():.2f} ms")

        self.display_timer = current_time