import argparse

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to video file or image. 'cam' for capturing video stream from camera",
        required=True,
        type=str)
    parser.add_argument(
        "-m_fc",
        "--model_face",
        help="Optional. Path to an .xml file with a trained Age/Gender Recognition model.",
        type=str,
        default=None)
    parser.add_argument(
        "-m_ag",
        "--model_age_gender",
        help="Optional. Path to an .xml file with a trained Age/Gender Recognition model.",
        type=str,
        default=None)
    parser.add_argument(
        "-l",
        "--cpu_extension",
        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.",
        type=str,
        default=None)
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target device for MobileNet-SSSD / Face Detection to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'FPGA', 'MYRIAD'],
        type=str)
    parser.add_argument(
        "-d_ag",
        "--device_age_gender",
        help="Specify the target device for Age/Gender Recognition to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default="CPU",
        choices=['CPU', 'GPU', 'FPGA', 'MYRIAD'],
        type=str)
    parser.add_argument(
        "-pp",
        "--plugin_dir",
        help="Path to a plugin folder",
        type=str,
        default=None)
    parser.add_argument(
        "--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        help="Probability threshold for object detections filtering",
        default=0.3,
        type=float)
    parser.add_argument(
        "-ptf",
        "--prob_threshold_face",
        help="Probability threshold for face detections filtering",
        default=0.5,
        type=float)
    parser.add_argument(
        '--no_v4l',
        help='cv2.VideoCapture without cv2.CAP_V4L',
        action='store_true')

    return parser