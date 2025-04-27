import subprocess

def run_single_eval(a1, a2):
    completed_process = subprocess.run(
        ["python", "evaluation.py", "--test-paper-title", str(a1), "--test-paper-abstract", str(a2)],
        capture_output=True,  # Capture stdout
        text=True             # Get output as string, not bytes
    )

    # Get the result (it will be in stdout)
    result = completed_process.stdout.strip()
    return result

if __name__ == "__main__":
    output = run_single_eval('t1', 'a1')
    print(f"Result from evaluation.py: \n{output}")
    # assign_marks_to_preds(output)
