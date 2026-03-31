from captioning import Captioning
from segment_feature import SegmentFeature
from tracking import Tracking
from reid import ReID
import os
import shutil
import ast
import sys
import re
from io import StringIO
from tools import ToolKit
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_openai import ChatOpenAI
from threading import Lock
from langchain_core.exceptions import OutputParserException
import time
import pickle
import traceback


CACHE_FILE = "react_prompt_cache.pkl"


class SandboxManager:


    def __init__(self, base_dir='./sandboxes', show_tracking=False, tracking_fps=15, sample_num=5):
        self.base_dir = base_dir
        self.show_tracking = show_tracking
        self.tracking_fps = tracking_fps
        self.sample_num = sample_num

        # Initialize processing components
        self.captioning = Captioning(video_path_list=[], base_dir=self.base_dir)
        self.temporal_feature = SegmentFeature(video_path_list=[], base_dir=self.base_dir)
        self.tracking = Tracking(
            video_path_list=[],
            base_dir=self.base_dir,
            tracking_fps=self.tracking_fps,
            sample_num=self.sample_num,
            show=self.show_tracking
        )
        self.reid = ReID(video_path_list=[], base_dir=self.base_dir)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "failed to get openai key")
        self.preprocess_resource_lock = Lock()
        self.llava_lock = Lock()
        self.datastructures_lock = Lock()

        self.loaded_video = {}
        self.fork_count = {}
        self.toolkits = {}

        prompt = hub.pull("hwchase17/react")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(prompt, f)


    def create_sandbox(self, sandbox_id):
        """Create a sandbox directory for video processing."""
        sandbox_path = os.path.join(self.base_dir, sandbox_id)
        os.makedirs(sandbox_path, exist_ok=True)
        return sandbox_path
    
    def fork(self, sandbox_id):
        
        with self.datastructures_lock:
            sandbox_path = os.path.join(self.base_dir, sandbox_id)

            if not os.path.exists(sandbox_path):
                raise FileNotFoundError(f"Sandbox {sandbox_id} not found")

            st = time.perf_counter()
            if sandbox_id not in self.fork_count:
                self.fork_count[sandbox_id] = 0
            
            self.fork_count[sandbox_id] += 1

            forked_sandbox_id = f'{sandbox_id}_{self.fork_count[sandbox_id]}'
            forked_sandbox_path = os.path.join(self.base_dir, forked_sandbox_id)

            shutil.copytree(sandbox_path, forked_sandbox_path)
            et = time.perf_counter()
            print(f'Time taken to fork: {sandbox_id} to {forked_sandbox_path} is {et - st}')
            if sandbox_id in self.loaded_video:
                self.loaded_video[forked_sandbox_id] = self.loaded_video[sandbox_id]
        
        try:
            self.generate_toolkit(sandbox_id=forked_sandbox_id)
        except Exception as e:
            print(f'Could not generate toolkit for forked sandbox: {forked_sandbox_id}')

        return {"sandbox_id": forked_sandbox_id}


    def load_video_into_sandbox(self, video_name, sandbox_id):
        """Copy a video from sample_videos into the sandbox."""
        with self.datastructures_lock:
            source_path = os.path.join('path/to/train/EgoSchema/videos', video_name)
            self.loaded_video[sandbox_id] = video_name
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Video {video_name} not found in {source_path} directory")

            # destination follows the structure: base_dir/sandbox_id/video.mp4
            dest_dir = os.path.join(self.base_dir, sandbox_id)
            dest_path = os.path.join(dest_dir, "video.mp4")
            shutil.copy2(source_path, dest_path)

            return 'Successfully loaded video into the sandbox'


    def stop_sandbox(self, sandbox_id):
        """Stop and remove a sandbox directory."""

        with self.datastructures_lock:
            print(f'DELETING AND REMOVING SANDBOX: {sandbox_id}')
            sandbox_path = os.path.join(self.base_dir, sandbox_id)
            if not os.path.exists(sandbox_path):
                raise FileNotFoundError(f"Sandbox {sandbox_id} not found")

            if sandbox_id in self.toolkits:
                self.toolkits[sandbox_id].cleanup()
                del self.toolkits[sandbox_id]
            
            if sandbox_id in self.fork_count:
                del self.fork_count[sandbox_id]

            shutil.rmtree(sandbox_path)
            del self.loaded_video[sandbox_id]

            return True


    def sandbox_exists(self, sandbox_id):
        """Check if a sandbox exists."""
        sandbox_path = os.path.join(self.base_dir, sandbox_id)
        return os.path.exists(sandbox_path)

    def generate_toolkit(self, sandbox_id):
        base_dir = os.path.join(self.base_dir, sandbox_id)
        toolkit = ToolKit(video_path=os.path.join(base_dir, "video.mp4"), base_dir=base_dir, vqa_tool='videollava', use_reid=True, openai_api_key=self.openai_api_key, captioning=self.captioning)

        with self.datastructures_lock:
            self.toolkits[sandbox_id] = toolkit

    def preprocess(self, sandbox_id):
        """Preprocess video in the sandbox by building temporal and object memory."""

        # Acquire lock
        with self.preprocess_resource_lock:
            print(f'Acquired lock to preprocess {sandbox_id}')

            base_dir = os.path.join(self.base_dir, sandbox_id)
            video_path_list = [os.path.join(base_dir, "video.mp4")]

            # Directory containing preprocessed video folders (configurable)
            preprocessed_dir = './egoschema_cache'

            def check_has_been_preprocessed(video_path):
                """Check if all required preprocessing files exist."""
                base_name = os.path.basename(video_path).replace(".mp4", "")
                video_dir = os.path.join(base_dir, base_name)
                if not os.path.exists(video_dir):
                    return False
                files = os.listdir(video_dir)
                required_files = [
                    "captions.json",
                    "segment_textual_embedding.pkl",
                    "segment_visual_embedding.pkl",
                    "segment2id.json",
                    "tracking.pkl",
                    "reid.pkl",
                    "tid2clip.pkl",
                    "tid2dinov2.pkl",
                    "uid2clip.pkl",
                    "reid.mp4"
                ]
                for f in required_files:
                    if f not in files:
                        return False
                return True

            def copy_preprocessed_data(video_path):
                """Check if preprocessed data exists in preprocessed_dir and copy it."""
                base_name = os.path.basename(video_path).replace(".mp4", "")
                video_dir = os.path.join(base_dir, base_name)

                # Check if preprocessed directory exists and contains the video folder
                if os.path.exists(preprocessed_dir) and sandbox_id in self.loaded_video:
                    
                    preprocessed_video_dir = os.path.join(preprocessed_dir, self.loaded_video[sandbox_id].replace('.mp4', ''), 'video')

                    if os.path.exists(preprocessed_video_dir) and os.path.isdir(preprocessed_video_dir):
                        print(f'Found preprocessed data in {preprocessed_video_dir}, copying to {sandbox_id}')

                        # Create destination directory if it doesn't exist
                        os.makedirs(video_dir, exist_ok=True)

                        # Copy all files from the preprocessed video folder
                        st = time.perf_counter()
                        for item in os.listdir(preprocessed_video_dir):
                            src = os.path.join(preprocessed_video_dir, item)
                            dst = os.path.join(video_dir, item)
                            if os.path.isfile(src):
                                shutil.copy2(src, dst)
                            elif os.path.isdir(src):
                                shutil.copytree(src, dst, dirs_exist_ok=True)
                        et = time.perf_counter()
                        print(f'Successfully copied preprocessed data from {preprocessed_dir}/{base_name} in {et - st}')
                        return True
                return False

            for video_path in video_path_list:
                if not check_has_been_preprocessed(video_path):
                    copy_preprocessed_data(video_path)

            preprocess_list = []
            for video_path in video_path_list:
                if not check_has_been_preprocessed(video_path):
                    preprocess_list.append(video_path)
                else:
                    self.generate_toolkit(sandbox_id=sandbox_id)

            if len(preprocess_list) == 0:
                return

            self.captioning.video_path_list = preprocess_list
            self.captioning.base_dir = base_dir
            self.captioning.run()

            self.temporal_feature.video_path_list = preprocess_list
            self.temporal_feature.base_dir = base_dir
            self.temporal_feature.run()

            # build object memory
            self.tracking.video_path_list = preprocess_list
            self.tracking.base_dir = base_dir
            self.tracking.run()

            self.reid.video_path_list = preprocess_list
            self.reid.base_dir = base_dir
            self.reid.run()

            self.generate_toolkit(sandbox_id=sandbox_id)

    def get_toolkit(self, sandbox_id):
        with self.datastructures_lock:
            return self.toolkits[sandbox_id]

    def object_memory_querying(self, sandbox_id, question):
        """Given a question about open-vocabulary objects such as 'how many people are there in the video?' or 'In which segments did the brown dog appear?', this tool will give the answer based on the object memory."""
        
        toolkit = self.get_toolkit(sandbox_id)

        @tool
        def database_querying(program):
            """given a MySQL program, this tool will query the database and return the results."""
            ans = toolkit.query_database(program=program)
            return '\n'+ans+'\n'
        @tool
        def open_vocabulary_object_retrieval(description):
            """given an open-vocabulary description of an object or a person (frying pan, person in red clothes e.g.), this tool will return the possible candidate object IDs that satisfy the description."""
            ans = toolkit.retrieve_candidate_objects(description=description)
            return '\n'+ans+'\n'
        
        # prompt = hub.pull("hwchase17/react")
        with open(CACHE_FILE, 'rb') as f:
            prompt = pickle.load(f)
        
        with open('prompts/database_query_prompt.txt') as f:
            t = f.read()
        
        prompt.template = t

        llm = ChatOpenAI(model='gpt-4.1', temperature=0.0, base_url="")
        tools = [database_querying, open_vocabulary_object_retrieval]
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=False)

        # original_stdout = sys.stdout
        # output_catcher = StringIO()
        # sys.stdout = output_catcher

        try:
            # try:
            #     output = agent_executor.invoke({"input": question})
                

            # except Exception as e:
            #     error_msg = str(e)
            #     if "Could not parse LLM output:" in error_msg:
            #         llm_output = error_msg.split("Could not parse LLM output:")[-1].strip()
            #         print(f"LLM responded with unparseable output: {llm_output}", file=original_stdout)
            #         return llm_output
            #     else:
            #         import traceback
            #         return f'Failed to call object_memory_querying: {traceback.format_exc()}'

            # output_catcher.seek(0)
            # lines = output_catcher.readlines()
            # color_pattern = re.compile(r'\x1B\[[0-9;]*[mK]')
            # answer = None
            # for line in lines:
            #     print(line, file=original_stdout, end='')
            #     if line.startswith("Final Answer: "):
            #         line = color_pattern.sub('', line)
            #         line = line.replace("Final Answer: ", "")
            #         answer = line
            # return answer
            output = agent_executor.invoke({"input": question})
            return output['output']
        except Exception as e:
            print(traceback.format_exc())
            return "I cannot answer that question"
        # finally:
        #     sys.stdout = original_stdout


    def segment_localization(self, sandbox_id, description):
        """Given a textual description, this tool will return the top-5 candidate segments that are most relevant to the description."""
        toolkit = self.get_toolkit(sandbox_id)
        answer = toolkit.segment_localization(description, k=5)
        return '\n'+answer+'\n'

    def log(self, line):
        with open('sandbox.log', 'a') as log_file:
            log_file.write(line)
            log_file.write('\n')

    def caption_retrieval(self, sandbox_id, input_tuple):
        """given an input tuple (start_segment_ID, end_segment_ID), this tool will retrieve all the captions between the two segments, 15 captions at most. end_segment_ID < start_segment_ID + 15."""
        
        input_tuple = ast.literal_eval(input_tuple)
        toolkit = self.get_toolkit(sandbox_id)
        if len(input_tuple) != 2:
            return "\nInvalid input tuple!\n"
        answer = toolkit.caption_retrieval(int(input_tuple[0]), int(input_tuple[1]))
        return '\n'+answer+'\n'


    def visual_question_answering(self, sandbox_id, input_tuple):
        """Given an input tuple (question, segment_ID), this tool will focus on the video segments starting from segment_ID-1 to segment_ID+1. It will return the description of the video segment and the answer to the question based on the segment."""

        self.llava_lock.acquire()

        input_tuple = ast.literal_eval(input_tuple)
        toolkit = self.get_toolkit(sandbox_id)
        if len(input_tuple) != 2:
            return "\nInvalid input tuple!\n"
        question = input_tuple[0]
        segment_id = int(input_tuple[1])
        print(f'Visual question answering on {question} and segment {segment_id}')
        answer = toolkit.visual_question_answering(question, segment_id)

        self.llava_lock.release()

        return '\n'+answer+'\n'

# Need thread safety only for preprocess and llava access
