import cv2
import numpy as np
from models.ocr import collect_textual_data_for_frame

POSITIVE = 1
NEGATIVE = -1
UNKNOWN = 0

MAX_SAMPLED = 10


def is_video_file(filename):
    video_file_extensions = (
        '.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.89', '.aaf', '.aec', '.aep', '.aepx',
        '.aet', '.aetx', '.ajp', '.ale', '.am', '.amc', '.amv', '.amx', '.anim', '.aqt', '.arcut', '.arf', '.asf', '.asx', '.avb',
        '.avc', '.avd', '.avi', '.avp', '.avs', '.avs', '.avv', '.axm', '.bdm', '.bdmv', '.bdt2', '.bdt3', '.bik', '.bin', '.bix',
        '.bmk', '.bnp', '.box', '.bs4', '.bsf', '.bvr', '.byu', '.camproj', '.camrec', '.camv', '.ced', '.cel', '.cine', '.cip',
        '.clpi', '.cmmp', '.cmmtpl', '.cmproj', '.cmrec', '.cpi', '.cst', '.cvc', '.cx3', '.d2v', '.d3v', '.dat', '.dav', '.dce',
        '.dck', '.dcr', '.dcr', '.ddat', '.dif', '.dir', '.divx', '.dlx', '.dmb', '.dmsd', '.dmsd3d', '.dmsm', '.dmsm3d', '.dmss',
        '.dmx', '.dnc', '.dpa', '.dpg', '.dream', '.dsy', '.dv', '.dv-avi', '.dv4', '.dvdmedia', '.dvr', '.dvr-ms', '.dvx', '.dxr',
        '.dzm', '.dzp', '.dzt', '.edl', '.evo', '.eye', '.ezt', '.f4p', '.f4v', '.fbr', '.fbr', '.fbz', '.fcp', '.fcproject',
        '.ffd', '.flc', '.flh', '.fli', '.flv', '.flx', '.gfp', '.gl', '.gom', '.grasp', '.gts', '.gvi', '.gvp', '.h264', '.hdmov',
        '.hkm', '.ifo', '.imovieproj', '.imovieproject', '.ircp', '.irf', '.ism', '.ismc', '.ismv', '.iva', '.ivf', '.ivr', '.ivs',
        '.izz', '.izzy', '.jss', '.jts', '.jtv', '.k3g', '.kmv', '.ktn', '.lrec', '.lsf', '.lsx', '.m15', '.m1pg', '.m1v', '.m21',
        '.m21', '.m2a', '.m2p', '.m2t', '.m2ts', '.m2v', '.m4e', '.m4u', '.m4v', '.m75', '.mani', '.meta', '.mgv', '.mj2', '.mjp',
        '.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
        '.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2', '.mpgindex', '.mpl',
        '.mpl', '.mpls', '.mpsub', '.mpv', '.mpv2', '.mqv', '.msdvd', '.mse', '.msh', '.mswmm', '.mts', '.mtv', '.mvb', '.mvc',
        '.mvd', '.mve', '.mvex', '.mvp', '.mvp', '.mvy', '.mxf', '.mxv', '.mys', '.ncor', '.nsv', '.nut', '.nuv', '.nvc', '.ogm',
        '.ogv', '.ogx', '.osp', '.otrkey', '.pac', '.par', '.pds', '.pgi', '.photoshow', '.piv', '.pjs', '.playlist', '.plproj',
        '.pmf', '.pmv', '.pns', '.ppj', '.prel', '.pro', '.prproj', '.prtl', '.psb', '.psh', '.pssd', '.pva', '.pvr', '.pxv',
        '.qt', '.qtch', '.qtindex', '.qtl', '.qtm', '.qtz', '.r3d', '.rcd', '.rcproject', '.rdb', '.rec', '.rm', '.rmd', '.rmd',
        '.rmp', '.rms', '.rmv', '.rmvb', '.roq', '.rp', '.rsx', '.rts', '.rts', '.rum', '.rv', '.rvid', '.rvl', '.sbk', '.sbt',
        '.scc', '.scm', '.scm', '.scn', '.screenflow', '.sec', '.sedprj', '.seq', '.sfd', '.sfvidcap', '.siv', '.smi', '.smi',
        '.smil', '.smk', '.sml', '.smv', '.spl', '.sqz', '.srt', '.ssf', '.ssm', '.stl', '.str', '.stx', '.svi', '.swf', '.swi',
        '.swt', '.tda3mt', '.tdx', '.thp', '.tivo', '.tix', '.tod', '.tp', '.tp0', '.tpd', '.tpr', '.trp', '.ts', '.tsp', '.ttxt',
        '.tvs', '.usf', '.usm', '.vc1', '.vcpf', '.vcr', '.vcv', '.vdo', '.vdr', '.vdx', '.veg','.vem', '.vep', '.vf', '.vft',
        '.vfw', '.vfz', '.vgz', '.vid', '.video', '.viewlet', '.viv', '.vivo', '.vlab', '.vob', '.vp3', '.vp6', '.vp7', '.vpj',
        '.vro', '.vs4', '.vse', '.vsp', '.w32', '.wcp', '.webm', '.wlmp', '.wm', '.wmd', '.wmmp', '.wmv', '.wmx', '.wot', '.wp3',
        '.wpl', '.wtv', '.wve', '.wvx', '.xej', '.xel', '.xesc', '.xfl', '.xlmv', '.xmv', '.xvid', '.y4m', '.yog', '.yuv', '.zeg',
        '.zm1', '.zm2', '.zm3', '.zmv')

    if filename.endswith(video_file_extensions):
        return True


def calc_shot_textual_data(shot):
    i = 1
    cap = cv2.VideoCapture(shot.title)
    total = 0
    sampled = 0
    while True:
        # Capture frame-by-frame
        ret, orig_frame = cap.read()
        if ret == False:
            break
        if sampled == MAX_SAMPLED:
            # Our operations on the frame come here
            frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
            td = collect_textual_data_for_frame(frame)
            words_num = td.words_num
            total += words_num
            i += 1
            sampled = 0
        sampled += 1
    if len == 0:
        return 0
    else:
        return total/i

    
def edge_based_difference(img1, img2):
    tau = 0.1
    N = 8
    win_w = int(img1.shape[0]/N)
    win_h = int(img1.shape[1]/N)
    i = 0
    changes = []
    # Crop out the window and process
    for r in range(0, img1.shape[0], win_w):
        for c in range(0, img1.shape[1] - win_h, win_h):
            window1 = img1[r:r + win_w, c:c + win_h]
            edges1 = cv2.Canny(window1, 100, 200)
            window2 = img2[r:r + win_w, c:c + win_h]
            edges2 = cv2.Canny(window2, 100, 200)
            d = np.linalg.norm(edges2 - edges1)
            changes.append(d)
            if d > tau:
                i += 1
    return changes, i


class Shot:
    def __init__(self, t, shot_title):
        self.starting_time = t
        self.ending_time = 0
        self.frames_arr = list()
        self.shot_stability = 0
        self.shot_stability_total = 0
        self.face_score = 0
        self.consistency = 0
        self.shot_len = 0
        self.title = shot_title
        self.f_t = 0
        self.shot_type = UNKNOWN

    def add_frame(self, img):
        self.frames_arr.append(img)

    def shot_stability_calc(self):
        shot_size = len(self.frames_arr)
        f_s = 0
        frames_edges_changes = []
        for k in range(1, shot_size):
            curr = self.frames_arr[k]
            curr = curr.astype(np.uint8)
            if k == 1:
                prev = np.zeros(curr.shape)
            else:
                prev = self.frames_arr[k-1]
            prev = prev.astype(np.uint8)
            changes, changed_blocks = edge_based_difference(prev, curr)
            med = np.median(changes)
            for n in range(0, len(changes)):
                var = changes[n] - med
                if var < 0:
                    var = 0
                changes[n] = var
            length = len(changes)
            changes = changes[5:length-5]
            frames_edges_changes.append(np.average(changes))
            f_s += changed_blocks

         # edge frames can be outlier
        total = sum(frames_edges_changes)/1000 # measure how the frames were changed
        f_s = f_s / (shot_size + 1)
        # print("sum: " + str(total))
        shot_stability_total = f_s # measure how many frames were changed
        # print('shot stability: ' + str(shot_stability_total))
        if f_s != 0:
            # print('ratio: ' + str(total/f_s))
            self.consistency = total/shot_stability_total
        self.shot_stability = frames_edges_changes.count(0)
        self.shot_stability_total = shot_stability_total
        self.frames_arr.clear()
        return self.shot_stability

    def calculate_len(self):
        cap = cv2.VideoCapture(self.title)
        fps = cap.get(cv2.CAP_PROP_FPS)
        len = 0
        while True:
            ret, orig_frame = cap.read()
            if ret == False:
                break
            len += 1
        self.shot_len = len
        self.ending_time = len/fps
        return len

    def classify_shot(self, theta_l, theta_s, theta_c, theta_ct):
        length = self.calculate_len()
        if length > theta_l:
            if self.shot_stability > theta_s or self.shot_stability_total < theta_s or \
                    (self.consistency < theta_c and self.consistency * self.shot_stability_total < theta_ct):
                self.shot_type = POSITIVE
            else:
                self.shot_type = NEGATIVE
        else:
            self.shot_type = UNKNOWN







