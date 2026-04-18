import os
import uuid
import threading
import time
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from config import Config
from utils.cleanup import CleanupManager
from utils.video_segmenter import VideoSegmenter
from utils.frame_extractor import FrameExtractor
from utils.frame_filter import FrameFilter
from utils.description_generator import DescriptionGenerator
from utils.matcher import HighlightMatcher
from utils.video_processor import VideoProcessor

app = Flask(__name__)
app.config.from_object(Config)

cleanup_manager = CleanupManager()
video_segmenter = VideoSegmenter(clip_duration=Config.CLIP_DURATION)
frame_extractor = FrameExtractor()
frame_filter = FrameFilter()
description_generator = DescriptionGenerator()
highlight_matcher = HighlightMatcher()
video_processor = VideoProcessor()

processing_jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def process_video(job_id):
    try:
        start_time = time.time()
        job = processing_jobs[job_id]

        print(f"✂️ Segmenting video for job {job_id}")
        clips_info = video_segmenter.segment_video(
            job['video_path'],
            job['clips_folder']
        )
        job['clips_info'] = clips_info
        job['clips_count'] = len(clips_info)

        if not clips_info:
            job['status'] = 'failed'
            job['error'] = 'No clips could be created from the uploaded video.'
            return

        total_extracted_frames = 0
        total_filtered_frames = 0
        total_deleted_frames = 0

        # Process clip by clip
        for clip_info in sorted(clips_info, key=lambda x: x["clip_index"]):
            print(f"🚀 Processing {clip_info['clip_id']}")

            # 1. extract all frames for this clip
            clip_frames = frame_extractor.extract_frames_from_clip(
                clip_info,
                job['frames_folder']
            )
            total_extracted_frames += len(clip_frames)

            if not clip_frames:
                print(f"⚠️ No frames extracted for {clip_info['clip_id']}")
                continue

            # 2. filter + delete redundant frame files
            selected_frames, deleted_count = frame_filter.filter_and_cleanup_clip(clip_frames)
            total_deleted_frames += deleted_count
            total_filtered_frames += len(selected_frames)

            if not selected_frames:
                print(f"⚠️ No selected frames for {clip_info['clip_id']}")
                continue

            print(f"✅ {clip_info['clip_id']} | extracted={len(clip_frames)} | selected={len(selected_frames)} | deleted={deleted_count}")

            # 3. one batch model request per clip
            description_generator.process_clip_batch(
                clip_info=clip_info,
                selected_frames=selected_frames,
                description_file=job['descriptions_file'],
                memory_file=job['memory_file']
            )

        job['frames_count'] = total_extracted_frames
        job['filtered_frames_count'] = total_filtered_frames
        job['deleted_frames_count'] = total_deleted_frames

        if total_filtered_frames == 0:
            job['status'] = 'failed'
            job['error'] = 'No useful frames remained after processing all clips.'
            return

        print(f"🎯 Matching highlights for job {job_id}")
        matches = highlight_matcher.match_highlights(
            descriptions_file=job['descriptions_file'],
            memory_file=job['memory_file'],
            user_prompt=job['prompt'],
            output_file=job['timestamps_file']
        )

        if not matches:
            job['status'] = 'failed'
            job['error'] = 'No matches found for your prompt. Try a different description.'
            return

        print(f"🎬 Creating highlight video for job {job_id}")
        output_path = video_processor.create_highlight_video(
            job['video_path'],
            job['timestamps_file'],
            job['output_video']
        )

        processing_time = time.time() - start_time

        job['status'] = 'completed'
        job['output_path'] = output_path
        job['matches_count'] = len(matches)
        job['highlights'] = matches
        job['processing_time'] = f"{processing_time:.2f} seconds"

        print(f"✅ Job {job_id} completed successfully in {processing_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error processing job {job_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['error'] = str(e)


@app.route('/')
def index():
    error = request.args.get('error')
    return render_template('index.html', error=error)


@app.route('/result/<job_id>')
def result_page(job_id):
    if job_id not in processing_jobs:
        return render_template('error.html', error="Job not found. Please upload a video first."), 404

    job = processing_jobs[job_id]

    return render_template(
        'result.html',
        job_id=job_id,
        status=job['status'],
        prompt=job.get('prompt', ''),
        matches_count=job.get('matches_count', 0) if job['status'] == 'completed' else 0,
        processing_time=job.get('processing_time', '') if job['status'] == 'completed' else '',
        highlights=job.get('highlights', []) if job['status'] == 'completed' else []
    )


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        prompt = request.form.get('prompt', '')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use mp4, avi, mov, mkv, or webm'}), 400

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > Config.MAX_FILE_SIZE:
            return jsonify({'error': f'File size exceeds {Config.MAX_FILE_SIZE // (1024 * 1024)}MB limit'}), 400

        cleanup_manager.clear_all()

        job_id = str(uuid.uuid4())

        filename = secure_filename(file.filename)
        video_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(video_path)

        job_frames_folder = os.path.join(Config.FRAMES_FOLDER, job_id)
        job_output_folder = os.path.join(Config.OUTPUT_FOLDER, job_id)
        job_clips_folder = os.path.join(Config.CLIPS_FOLDER, job_id)

        os.makedirs(job_frames_folder, exist_ok=True)
        os.makedirs(job_output_folder, exist_ok=True)
        os.makedirs(job_clips_folder, exist_ok=True)

        processing_jobs[job_id] = {
            'status': 'processing',
            'video_path': video_path,
            'prompt': prompt,
            'frames_folder': job_frames_folder,
            'output_folder': job_output_folder,
            'clips_folder': job_clips_folder,
            'descriptions_file': os.path.join(job_output_folder, Config.DESCRIPTION_JSON_NAME),
            'memory_file': os.path.join(job_output_folder, Config.MEMORY_JSON_NAME),
            'timestamps_file': os.path.join(job_output_folder, Config.HIGHLIGHTS_XLSX_NAME),
            'output_video': os.path.join(job_output_folder, Config.OUTPUT_VIDEO_NAME),
            'created_at': time.time(),
            'clips_info': [],
            'clips_count': 0,
            'frames_count': 0,
            'filtered_frames_count': 0,
            'deleted_frames_count': 0
        }

        thread = threading.Thread(target=process_video, args=(job_id,))
        thread.daemon = True
        thread.start()

        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'redirect_url': url_for('result_page', job_id=job_id),
            'message': 'Video uploaded and processing started'
        }), 202

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]
    response = {
        'status': job['status'],
        'clips_count': job.get('clips_count', 0),
        'frames_count': job.get('frames_count', 0),
        'filtered_frames_count': job.get('filtered_frames_count', 0),
        'deleted_frames_count': job.get('deleted_frames_count', 0)
    }

    if job['status'] == 'completed':
        response['download_url'] = url_for('download_video', job_id=job_id)
        response['preview_url'] = url_for('preview_video', job_id=job_id)
        response['matches_count'] = job.get('matches_count', 0)
        response['processing_time'] = job.get('processing_time', 'Unknown')
        response['highlights'] = job.get('highlights', [])
    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')

    return jsonify(response)


@app.route('/api/highlights/<job_id>')
def get_highlights(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]

    if job['status'] == 'completed':
        return jsonify({
            'highlights': job.get('highlights', []),
            'prompt': job.get('prompt', ''),
            'matches_count': job.get('matches_count', 0),
            'clips_count': job.get('clips_count', 0),
            'frames_count': job.get('frames_count', 0),
            'filtered_frames_count': job.get('filtered_frames_count', 0),
            'deleted_frames_count': job.get('deleted_frames_count', 0)
        })

    return jsonify({'error': 'Video not ready yet'}), 400


@app.route('/download/<job_id>')
def download_video(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]

    if job['status'] != 'completed':
        return jsonify({'error': 'Video not ready yet'}), 400

    if not os.path.exists(job['output_path']):
        return jsonify({'error': 'Output file not found'}), 404

    return send_file(
        job['output_path'],
        as_attachment=True,
        download_name='highlight_video.mp4',
        mimetype='video/mp4'
    )


@app.route('/preview/<job_id>')
def preview_video(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]

    if job['status'] != 'completed':
        return jsonify({'error': 'Video not ready yet'}), 400

    if not os.path.exists(job['output_path']):
        return jsonify({'error': 'Output file not found'}), 404

    return send_file(
        job['output_path'],
        mimetype='video/mp4'
    )


@app.route('/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = processing_jobs[job_id]

    try:
        if os.path.exists(job['video_path']):
            os.remove(job['video_path'])

        if os.path.exists(job['frames_folder']):
            import shutil
            shutil.rmtree(job['frames_folder'])

        if os.path.exists(job['clips_folder']):
            import shutil
            shutil.rmtree(job['clips_folder'])

        if os.path.exists(job['output_folder']):
            import shutil
            shutil.rmtree(job['output_folder'])

        del processing_jobs[job_id]

        return jsonify({'message': 'Job cleaned up successfully'})

    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


@app.route('/jobs')
def list_jobs():
    jobs_info = {}
    for job_id, job in processing_jobs.items():
        jobs_info[job_id] = {
            'status': job['status'],
            'prompt': job['prompt'][:50] + '...' if len(job['prompt']) > 50 else job['prompt'],
            'created_at': job.get('created_at', 0),
            'matches_count': job.get('matches_count', 0),
            'clips_count': job.get('clips_count', 0),
            'frames_count': job.get('frames_count', 0),
            'filtered_frames_count': job.get('filtered_frames_count', 0),
            'deleted_frames_count': job.get('deleted_frames_count', 0)
        }
    return jsonify(jobs_info)


if __name__ == '__main__':
    print("=" * 50)
    print("🎬 Video Highlight Extractor")
    print("=" * 50)
    print(f"📍 Server running at: http://localhost:5000")
    print(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"✂️ Clips folder: {Config.CLIPS_FOLDER}")
    print(f"🖼️ Frames folder: {Config.FRAMES_FOLDER}")
    print(f"📹 Output folder: {Config.OUTPUT_FOLDER}")
    print("=" * 50)
    print("⚠️  Press CTRL+C to stop the server")
    print("=" * 50)

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)