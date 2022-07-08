from pathlib import Path
import shutil

pheme5 = Path('./data/pheme-rnr-dataset')
pheme9 = Path('./data/all-rnr-annotated-threads')

dir_pheme5 = [e for e in pheme5.iterdir() if e.is_dir()]
dir_pheme9 = [e for e in pheme9.iterdir() if e.is_dir()]

for event in dir_pheme5:
    event_name = event.name
    print( event_name )
    # pheme5_event_path = event / 
    pheme9_event_path = pheme9 / f"{event_name}-all-rnr-threads"
    # print(pheme9_event_path)
    labels = [e for e in event.iterdir() if e.is_dir()]
    for label in labels:
        pheme9_label_path = pheme9_event_path / label.name
        # print(pheme9_label_path)
        threads = [e for e in label.iterdir() if e.is_dir()]
        for thread in threads:
#             pheme9_thread_path = pheme9_label_path / thread.name / "structure.json"
#             # pheme5_thread_path = thread
#             # print(pheme9_thread_path)
#             shutil.copy(pheme9_thread_path, thread)
            
            pheme9_thread_path = pheme9_label_path / thread.name / "annotation.json"
            # pheme5_thread_path = thread
            # print(pheme9_thread_path)
            shutil.copy(pheme9_thread_path, thread)

