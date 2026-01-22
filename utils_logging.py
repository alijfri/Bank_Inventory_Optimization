import sys
import os
from datetime import datetime
from inputs import optimization_policy, rdp_to_be_solved,opt_start_date, opt_finish_date,sim_weeks,opt_weeks
class Tee:
    def __init__(self, stream, filename):
        self.stream = stream
        self.file = open(filename, "a", buffering=1, encoding="utf-8")
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()
def build_run_tag():
    #rdp_tag = "-".join(map(str, rdp_to_be_solved))
    return (
        f"{optimization_policy}_"
        f"{rdp_to_be_solved}_"
        f"{opt_start_date}_to_{opt_finish_date}_"
        f"simW={sim_weeks}_"
        f"optW={opt_weeks}"
    )

# def setup_logging(log_dir="logs"):
#     os.makedirs(log_dir, exist_ok=True)
#     log_path = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
#     sys.stdout = Tee(sys.stdout, log_path)
#     sys.stderr = Tee(sys.stderr, log_path)
    
#     print(f"Logging to {log_path}", flush=True)
#     return log_path
def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    run_tag = build_run_tag()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_path = os.path.join(
        log_dir,
        f"{run_tag}_{timestamp}.log"
    )

    sys.stdout = Tee(sys.stdout, log_path)
    sys.stderr = Tee(sys.stderr, log_path)

    print(f"Logging to {log_path}", flush=True)
    return log_path
