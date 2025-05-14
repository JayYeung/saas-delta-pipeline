from __future__ import annotations
import json, time
import pandas as pd
import cv2
import os
import base64
from openai import OpenAI
from llm_confidence.logprobs_handler import LogprobsHandler
from moviepy import VideoFileClip


# ---------------------------------------------------------------------------- helpers
# def prepare_skill_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Prepare skill definition dataframe for consistent processing.
    
#     Args:
#         df: Raw skill definitions dataframe
        
#     Returns:
#         Processed dataframe with:
#         - Cleaned column names (stripped whitespace)
#         - Guaranteed "Skill" column (renames first column if needed)
#         - Reset 0-based integer index (used as reference IDs)
        
#     Note:
#         The reset index becomes the canonical ID system used throughout labeling
#     """
#     df = df.copy()
#     df.columns = df.columns.str.strip()
#     if "Skill" not in df.columns:
#         df.rename(columns={df.columns[0]: "Skill"}, inplace=True)
#     return df.reset_index(drop=True)  # Ensure 0-based integer IDs

class VideoSummarizer:
    def __init__(self, client, image_model='gpt-4o-mini', audio_model='whisper-1'):
        self.client = client
        self.image_model = image_model
        self.audio_model = audio_model

    def get_sampled_video_frames(self, videos_path, max_frames=20):
        video = cv2.VideoCapture(videos_path)
        base64_frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        video.release()

        step = max(1, len(base64_frames) // max_frames)
        return base64_frames[::step]

    # --- Prompt builders ---
    def _build_system_prompt(self) -> str:
        return f"""Give a short summary of the contents of the delta training video. 
        Pay attention for the skills learned and proficiency levels."""

    def summarize_video(self, video_path):
        frames = self.get_sampled_video_frames(video_path)
        system_prompt = self._build_system_prompt()
        user_prompt = f"Summarize: {os.path.basename(video_path)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    user_prompt,
                    *[{"image": img, "resize": 768} for img in frames],
                ],
            },
        ]

        response = self.client.chat.completions.create(model=self.image_model, messages=messages)
        raw_output = response.choices[0].message.content.strip()

        return raw_output
    
    def transcribe_video(self, video_path):
        # Create a temporary audio file
        temp_audio_path = "temp_audio.mp3"
        
        # Extract audio from MP4
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio_path)
        video.close()
        
        # Use the temporary audio file
        with open(temp_audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=self.audio_model,  # or "gpt-4o-mini-audio" if available
                file=audio_file,
                response_format="text"
            )
        
        # Delete the temporary file
        os.remove(temp_audio_path)
    
        return transcription
    


    # --- Main function to walk folder and save to JSON ---
    def summarize_all_videos(self, videos_path, output_path):
        dataset = {"data": []}
        video_files = [f for f in os.listdir(videos_path) if f.endswith(".mp4")]

        for filename in video_files:
            video_path = os.path.join(videos_path, filename)
            print(f"Processing {filename}...")
            try:
                transcription = self.transcribe_video(video_path)
            except Exception as e:
                print(f"Failed on {filename}: {e}")
                transcription = "None"
            print(f"Processing {filename}...")
            try:
                image_summary = self.summarize_video(video_path)
            except Exception as e:
                print(f"Failed on {filename}: {e}")
                image_summary = "None"

            dataset["data"].append({
                "title": filename,
                "video_path": video_path,
                "transcript": transcription,
                "image_summary": image_summary
            })

            # Save after each to prevent loss
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)

        print("All videos processed and saved to", output_path)

# ---------------------------------------------------------------------------- main class
class SummaryLabeler:
    """
    Automated skill labeling system for training content metadata.
    
    Attributes:
        client: OpenAI API client instance
        model: Name of LLM model to use (e.g., "gpt-4")
        max_retries: Maximum API call attempts before failing
        delay: Base delay between retry attempts (seconds)
        output_path: Path for incremental result saving
        data: Loaded content metadata to process
        skill_def: Processed skill definitions dataframe
        log_handler: Confidence score calculator
    """
    
    def __init__(
        self,
        skill_def: pd.DataFrame,
        proficiency_def: pd.DataFrame,
        *,
        client: OpenAI,
        model: str,
        max_retries: int = 3,
        delay: float = 1.0,
    ):
        """
        Initialize the text labeler with configuration and data sources.
        
        Args:
            data_path: Path to JSON file containing content metadata
            skill_def: Dataframe of skill definitions (must contain "Skill" column)
            client: Configured OpenAI client instance
            model: LLM model identifier to use
            max_retries: API call retry attempts (default: 3)
            delay: Base delay between retries in seconds (default: 1.0)
            output_path: Results output path (default: "resources/results.json")
        """
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.delay = delay
        self.skill_def = skill_def
        self.proficiency_def = proficiency_def
        self.log_handler = LogprobsHandler()

    # ---------------------------------------------------------------- prompt builders
    def _build_user_prompt_content(self, entry: dict) -> str:
        """
        Construct user prompt from content metadata entry.
        
        Args:
            entry: Dictionary containing content metadata with keys:
                   - title
                   - description 
                   - transcript
                   - image_summary
                   
        Returns:
            Concatenated string of relevant content fields
        """
        return "\n".join(
            f"{k}: {entry[k]}" for k in ("title", "transcript", "image_summary")
        )

    def _build_system_prompt_content(self):
            skill_options = '; '.join(
                f"(id: {i}, label: {row['Skill']})" for i, row in self.skill_def.iterrows()
            )
            proficiency_options = '; '.join(
                f"(code: {row['Proficiency Code']}, level: {row['Proficiency Level']})"
                for _, row in self.proficiency_def.iterrows()
            )
            return f"""You are labeling Delta airline training content. 
        Pick the most appropriate SKILL LABELS AND ASSOCIATE EACH with a PROFICIENCY LEVEL. Tend to choose lower profiency levels
        Choose ONLY from the following:

        Skills: {skill_options}
        Proficiency Levels: {proficiency_options}

        You must output 3–5 skill IDs with their proficiency code in DESCENDING ORDER OF RELEVANCE. 

        STRICT OUTPUT FORMAT (NO EXTRA TEXT): 
        Labels: <skill_id>:<proficiency_code>, <skill_id>:<proficiency_code>, ..."""

    # ---------------------------------------------------------------- single call
    def _build_labels_logprobs(self, entry: dict) -> tuple[list[str], dict[str, float]]:
        """
        Generate skill labels for a single content entry.
        
        Args:
            entry: Content metadata dictionary to label
            
        Returns:
            Tuple containing:
            - List of skill names (ordered by relevance)
            - Dictionary of confidence scores per label key
            
        Raises:
            OpenAI API exceptions after max retries exceeded
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt_content()},
                {"role": "user", "content": self._build_user_prompt_content(entry)},
            ],
            logprobs=True,
            top_logprobs=2,
            response_format={"type": "json_object"},
            temperature=0,
            stream=False,
        )

        reply_json = json.loads(resp.choices[0].message.content.strip())
        tokens = resp.choices[0].logprobs.content

        # Calculate confidence scores for each label
        logp_fmt = self.log_handler.format_logprobs(tokens)
        confidences = self.log_handler.process_logprobs(logp_fmt)  # {'label_1': p1, ...}

        # Convert numeric IDs to human-readable skill names
        label_ids = [reply_json[k] for k in sorted(reply_json)]  # label_1, label_2, ...
        skills = self.skill_def.loc[label_ids, "Skill"].tolist()

        return { 'Skill': skills, 'Confidence': confidences }
    
    def _build_labels_simple(self, entry: dict) -> tuple[list[str], dict[str, float]]:
        """
        Generate skill labels for a single content entry.
        
        Args:
            entry: Content metadata dictionary to label
            
        Returns:
            Tuple containing:
            - List of skill names (ordered by relevance)
            - Dictionary of confidence scores per label key
            
        Raises:
            OpenAI API exceptions after max retries exceeded
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt_content()},
                {"role": "user", "content": self._build_user_prompt_content(entry)},
            ],
            stream=False
        )

        raw_content = response.choices[0].message.content.strip()
        # print("Raw model response:", raw_content)

        label_section = raw_content[len("Labels:"):].strip()
        label_pairs = [pair.strip() for pair in label_section.split(',') if ':' in pair]

        skill_proficiency = []
        for pair in label_pairs:
            try:
                skill_id, prof_code = map(int, pair.split(':'))
                skill_row = self.skill_def.iloc[skill_id]
                prof_row = self.proficiency_def[self.proficiency_def['Proficiency Code'] == prof_code].iloc[0]

                skill_proficiency.append({
                    'Skill': skill_row['Skill'],
                    'Proficiency': prof_row['Proficiency Level'],
                    'Description': prof_row['Proficiency Description']
                })
            except Exception as e:
                print(f"Skipping malformed label pair '{pair}': {e}")
                continue

        return skill_proficiency

    # ---------------------------------------------------------------- bulk driver
    def revise_metadata_labels(self, data_path, logprobs=False) -> None:
        """
        Process all content entries to generate and save skill labels.
        
        Processes all entries in self.data['data'], adding:
        - "labels": List of skill names
        - "confidences": Per-label confidence scores
        
        Saves results incrementally to output_path.
        
        Note:
            Implements retry logic with exponential backoff
            Provides real-time console feedback
            Maintains results through interruptions
        """
        with open(data_path, "r") as f:
            self.data = json.load(f)

        for entry in self.data["data"]:
            print(f"\nGenerating labels for: {entry['title']}")
            for attempt in range(self.max_retries):
                try:
                    if logprobs:
                        new_labels = self._build_labels_logprobs(entry)
                    else:
                        new_labels = self._build_labels_simple(entry)
                except Exception as e:
                    print("Error:", e)
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.delay * (attempt + 1))
                else:
                    break

            entry["labeled_skills"] = new_labels
            # entry["confidences"] = conf

            # Incremental save for crash recovery
            with open(data_path, "w") as f:
                json.dump(self.data, f, indent=4)
            
            print("Added labels")
            # print("✓ Added labels:", labels)
            # print("  confidences :", conf)

        print("\nAll entries processed successfully.")


# ---------------------------------------------------------------------------- usage example
if __name__ == "__main__":
    skill_def_path = "resources/skill_definitions.csv"
    proficiency_def_path = "resources/Proficiency Levels (1).xlsx"
    videos_path = "resources/videos"
    output_path = "resources/final_metadata.json"
    if os.path.exists(output_path):
        raise Exception("Output path already exists.")
    # Example initialization and execution
    skill_def = pd.read_csv(skill_def_path)
    proficiency_def = pd.read_excel(proficiency_def_path)
    summary_labeler_client = OpenAI(api_key="sk-REDACTED", base_url="https://api.deepseek.com")
    video_summarizer_client = OpenAI(api_key="sk-REDACTED")
    
    labeler = SummaryLabeler(
        skill_def=skill_def,
        proficiency_def=proficiency_def,
        client=summary_labeler_client,
        model="deepseek-chat",
    )
    
    summarizer = VideoSummarizer(
        client=video_summarizer_client,
        image_model="gpt-4o-mini",
        audio_model="whisper-1",
    )

    summarizer.summarize_all_videos(videos_path, output_path)
    labeler.revise_metadata_labels(output_path)