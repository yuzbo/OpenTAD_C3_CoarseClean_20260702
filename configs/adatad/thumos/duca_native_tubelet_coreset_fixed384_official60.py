"""Task-state native-tubelet coreset; only anchor placement differs from control."""

_base_ = ["./duca_native_tubelet_uniform_reconstruct_fixed384_official60.py"]

native_tubelet_contract = dict(selection_policy="native_tubelet_coreset")

model = dict(
    frame_selector=dict(
        acquisition_policy="native_tubelet_coreset",
    ),
)
