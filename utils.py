class Shot:
    def __init__(self, t):
        self.starting_time = t
        self.ending_time = 0
        self.frames_arr = list()
        self.shot_stability = 0
        self.face_score = 0

    def add_frame(self, img):
        self.frames_arr.append(img)

    def update_shot_stability(self, shot_s):
        self.shot_stability = shot_s
        self.frames_arr.clear()