import numpy

mode_str = "MODE:\t\t"
emo_str = "EMOTION:\t\t"
recording_str = "RECORDING:\t"
clear_canvas = "Clear Screen"
new_start = "New Starting Position"
new_goal = "New Goal Position"
not_on_start_warning = "Error! Not close enough to the start. Please try again."
no_start_warning = "Error! No starting location defined. Please try again."
not_on_goal_warning = "Error! Not close enough to the goal. Please try again."
no_goal_warning = "Error! No goal location defined. Please try again."
save_demo_to_file = "Save demo to file?"
save_current_path = "Save Current Path"
current_path_empty= "The current Path is empty. Cannot Save or Replay."
replay_current_path = "Replay Current Path"
open_path_from_file = "Open Path from File"
path_from_file_empty = "The Path loaded from file is empty. Cannot Replay."
replay_path_from_file = "Replay Path from File"

# recording_active_color = "red"
# recording_inactive_color = "gray"

emotions = [    "Happy",
                        "Sad",
                        "Angry",
                        "Afraid",
                        "Disgusted",
                        "Surprise",
]

modes = [   "Demonstration",
                    "Playback",
                    "Training",
                    "Testing"
]

recording = [     "No",
                        "YES",
]
record = [  "Start Recording",
                 "Stop Recording",
]
recording_color = [    "gray",
                                "red",
                                "black",
]
# time_interval = 10 # 100 Hertz
time_interval = 50 # 50 Hertz
# time_interval = 100 # 10 Hertz
