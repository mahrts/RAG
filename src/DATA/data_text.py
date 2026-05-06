"""This scripts fetches the text data from url."""

import os
import requests
from pathlib import Path

base_url = "https://raw.githubusercontent.com/udacity/cd13318-exercises-project/main/Project-NASA-Mission-Intelligence-Starter/data_text/"

text_base_path = Path(__file__).resolve().parent.parent.parent / "data_text"

missions = {"apollo11": [
                          "19900066485_textract_full_text.txt",
                          "Apollo_11_Flight_Plan_HSK_textract_full_text.txt",
                          "NASA_NTRS_Archive_19710015566_textract_full_text.txt",
                          "a11transcript_pao_textract_full_text.txt",
                          "a11transcript_tec_textract_full_text.txt",
                          "a11transscript_cm_textract_full_text.txt"
                        ],
            "apollo13": [
                         "AS13_CM_textract_full_text.txt",
                         "AS13_PAO_textract_full_text.txt",
                         "AS13_TEC_textract_full_text.txt"
                        ],
            "challenger": [
                           "107-AAG_STS-51L_Mission_Audio_transcript.txt",
                           "108-AAG_STS-51L_Mission_Audio_transcript.txt",
                           "109-AAG_STS-51L_Mission_Audio_transcript.txt"
                          ]
            }
def get_data():
    """Download NASA mission .txt data to local path."""
    success = True  # assume success unless something fails

    try:
        if not os.path.isdir(text_base_path):
            os.mkdir(text_base_path)

        for mission in list(missions.keys()):

            if not os.path.isdir(text_base_path / mission):
                os.mkdir(text_base_path / mission)

            for file in missions[mission]:
                try:
                    file_url = base_url + mission + "/" + file
                    content = requests.get(file_url)
                    content.raise_for_status()
                    
                    #avoid duplicating text
                    file_path = text_base_path / mission / file
                    if file_path.exists():
                        file_path.unlink()

                    with open(file_path, "w", encoding="utf8") as f:
                        f.write(content.text)

                except Exception:
                    success = False

    except Exception:
        return False

    return success

if __name__ == "__main__":
    success = get_data()
    print(success)
