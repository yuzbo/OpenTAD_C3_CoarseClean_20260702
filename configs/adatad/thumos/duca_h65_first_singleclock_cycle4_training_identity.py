_base_ = ["./duca_h65_first_singleclock_cycle4.py"]

dataset = dict(test=dict(subset_name="training"))
evaluation = dict(subset="training")
