import subprocess
import numpy as np
import json

SHUTDOWN_TIMEOUT = 5 # seconds

class FFmDPegr:
    """
        Custom FFmpeg wrapper for decoding videos.
    """
    def __init__(self, video_path, start_frame=None, end_frame=None, pix_fmt='bgr24', batch_size=16):
        """
        Initialize the FFmpeg decoder.
        
        Parameters:
            video_path (str): Path to the video file.
            width (int): Width of the raw output frames.
            height (int): Height of the raw output frames.
            pix_fmt (str): Pixel format (e.g. 'rgb24').
            batch_size (int): Number of frames to decode per batch.
        """
        self.video_path = video_path
        self.pix_fmt = pix_fmt
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.batch_size = batch_size
        self.decoder_process = None
        self.extract_video_metadata(video_path)


    def extract_video_metadata(self, video_path):
        """
            Uses ffprobe to extract video metadata.
            
            Returns:
                width (int), height (int), fps (float), nb_frames (int or None), duration (float or None)
        """
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
            '-of', 'json', video_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        stream = metadata['streams'][0]

        self.width = int(stream['width'])
        self.height = int(stream['height'])

        # r_frame_rate is a string like "30000/1001". Evaluate it to get a float.
        numerator, denominator = stream['r_frame_rate'].split('/') 
        self.fps = float(numerator) / float(denominator)

        # nb_frames may not be available or might be 'N/A'
        try:
            self.total_frames = int(stream['nb_frames'])
        except (KeyError, ValueError):
            self.total_frames = None

        # Duration in seconds, if available.
        try:
            self.duration = float(stream['duration'])
        except (KeyError, ValueError):
            self.duration = None


    def get_metadata(self):
        return self.width, self.height, self.fps


    def metadata_to_string(self):
        str =   f"Width: {self.width}  Height: {self.height}\n"
        str +=  f"Total Frames: {self.total_frames}\n"
        str +=  f"FPS: {self.fps}\n"
        return str


    def split_frame(self, frame=None):
        # If no frame is passed, it is defaulted to the objects current frame
        left_frame = frame[:self.height//2, :]
        right_frame = frame[self.height//2:, :]
        return left_frame, right_frame


    def start(self):
        """
            Launch the FFmpeg subprocess to start decoding.
            Allows setting a start and end frame by converting them to time offsets.
        """
        # Build the command as a list.
        command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
        ]
        # If a start frame is specified, convert to start time (in seconds)
        if hasattr(self, 'start_frame') and self.start_frame is not None and self.start_frame != 0:
            start_time = self.start_frame / self.fps
            command.extend(['-ss', str(start_time)])
        
        # Input file
        command.extend(['-i', self.video_path])
        
        # If an end frame is specified, compute the duration.
        if hasattr(self, 'end_frame') and self.end_frame is not None:
            if hasattr(self, 'start_frame') and self.start_frame is not None and self.start_frame != 0:
                duration = (self.end_frame - self.start_frame) / self.fps
            else:
                duration = self.end_frame / self.fps
            command.extend(['-t', str(duration)])
        
        # Continue with the rest of the command.
        command.extend([
            '-f', 'rawvideo',
            '-pix_fmt', self.pix_fmt,
            '-vcodec', 'rawvideo',
            '-'  # Output to stdout
        ])
        
        # Start the process.
        self.decoder_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            bufsize=10**8
        )
        print("FFmDPegged process started.")


    def stop(self):
        """
            Terminate the FFmpeg process.
        """
        if self.decoder_process:
            try:
                # Signal FFmpeg to terminate gracefully.
                self.decoder_process.terminate()
                
                # Poll until the process has terminated or a timeout has been reached.
                import time
                timeout = SHUTDOWN_TIMEOUT
                t0 = time.time()
                while self.decoder_process.poll() is None and time.time() - t0 < timeout:
                    time.sleep(0.1)
                
                # If it's still not terminated, force kill it.
                if self.decoder_process.poll() is None:
                    self.decoder_process.kill()
            except Exception as e:
                print("Error stopping FFmDPegged process:", e)
            finally:
                try:
                    # Drain any remaining output (if non-blocking is possible) or close stdout.
                    self.decoder_process.stdout.close()
                except Exception as e:
                    print("Error closing stdout:", e)
                self.decoder_process.wait()
                self.decoder_process = None
                print("FFmDPegged process stopped.")


    def get_frame(self):
        """
            Read a batch of frames from the FFmpeg process.
            
            Returns:
                A NumPy array of shape (batch_size, height, width, 3) if successful,
                or None if no frames could be read.
        """
        if self.decoder_process is None:
            raise Exception("FFmpeg process not started. Call start() first.")

        frame_size = self.width * self.height * 3  # assuming 3 channels for rgb24
        raw_frame = self.decoder_process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            return  # End of stream

        return np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))


    def get_batch(self):
        """
            Read a batch of frames from the FFmpeg process.
            
            Returns:
                A NumPy array of shape (batch_size, height, width, 3) if successful,
                or None if no frames could be read.
        """
        if self.decoder_process is None:
            raise Exception("FFmpeg process not started. Call start() first.")

        frame_size = self.width * self.height * 3  # assuming 3 channels for rgb24
        frames = []
        for i in range(self.batch_size):
            raw_frame = self.decoder_process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break  # End of stream
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
            frames.append(frame)
        
        if not frames:
            return None
        return np.stack(frames)


class FFmPegr:
    """
        Custom FFmpeg wrapper for encoding videos.
    """
    def __init__(self, output_path, width, height, fps, encoder='libx264', input_pix_fmt='bgr24', output_pix_fmt='yuv420p'):
        self.output_path = output_path
        self.encoder = encoder
        self.input_pix_fmt = input_pix_fmt
        self.output_pix_fmt = output_pix_fmt
        self.width = width
        self.height = height
        self.fps = fps
        self.encoder_process = None


    def start(self):
        """
            Launch the FFmpeg subprocess for encoding.
            
            Parameters:
                output_video_path (str): Where to write the encoded video.
                encoder (str): The FFmpeg video encoder to use.
                output_pix_fmt (str): Pixel format for output video.
        """
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists.
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', self.input_pix_fmt,
            '-r', str(self.fps),
            '-i', '-',  # Input from stdin.
            '-an',  # No audio.
            '-vcodec', self.encoder,
            '-pix_fmt', self.output_pix_fmt,
            self.output_path
        ]
        self.encoder_process = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=10**8)
        print("FFmPegr encoder started.")


    def encode_frame(self, frame):
        """
            Write a single frame to the FFmpeg encoder process.
            
            Parameters:
                frame (np.ndarray): Frame in raw format (matching self.width, self.height, channels).
        """
        if self.encoder_process and self.encoder_process.stdin:
            self.encoder_process.stdin.write(frame.tobytes())


    def stop(self):
        """
            Terminate the FFmpeg encoder process gracefully.
        """
        if self.encoder_process:
            try:
                self.encoder_process.stdin.close()
            except Exception as e:
                print("Error closing encoder stdin:", e)
            self.encoder_process.wait()
            self.encoder_process = None
            print("FFmPegr encoder stopped.")


def test_dpegr(video_path):
    import cv2

    batch_size = 16

    dpegr = FFmDPegr(
        video_path,
        start_frame=0,
        end_frame=10,
        pix_fmt='bgr24',
        batch_size=batch_size
    )
    print(dpegr.metadata_to_string())

    dpegr.start_depegr()
    batch = dpegr.get_batch()

    if batch is not None:
        print("Decoded batch shape:", batch.shape)
        # For example, show the first frame using OpenCV:
        cv2.imshow("Frame", batch[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    dpegr.stop_depegr()

def test_pegr(frame_dir, output_path):
    import os, cv2

    fps = 30
    # Collect all image file paths (adjust extensions as needed)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    frame_files = sorted([
        os.path.join(frame_dir, f) 
        for f in os.listdir(frame_dir)
        if f.lower().endswith(valid_extensions)
    ])
    
    if not frame_files:
        print("No frame images found in", frame_dir)
        return

    # Read the first frame to get dimensions
    sample_frame = cv2.imread(frame_files[0])
    if sample_frame is None:
        print("Error reading the sample frame:", frame_files[0])
        return
    height, width, _ = sample_frame.shape

    pegr = FFmPegr(
        output_path,
        width, height, fps
    )
    pegr.start()

    for index, fpath in enumerate(frame_files):
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"Warning: Could not read frame {fpath}")
            continue
        pegr.encode_frame(frame)
        print(f"Encoded frame {index}/{len(frame_files)}", end="\r")
    
    pegr.stop()
    print("\nEncoding complete.")


# Example usage:
if __name__ == "__main__":
    frame_dir = ""
    output_path = ""
    test_pegr(frame_dir, output_path)
