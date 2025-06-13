import argparse
import os

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

from posture_analyzer import PostureAnalyzer


def extract_landmarks(pose_landmarks, frame_width, frame_height, mp_pose):
    """
    Extract key landmarks from MediaPipe pose results
    """
    lm = pose_landmarks
    lmPose = mp_pose.PoseLandmark
    landmarks = {}

    try:
        # Left shoulder
        landmarks["l_shoulder"] = (
            int(lm.landmark[lmPose.LEFT_SHOULDER].x * frame_width),
            int(lm.landmark[lmPose.LEFT_SHOULDER].y * frame_height),
        )

        # Right shoulder
        landmarks["r_shoulder"] = (
            int(lm.landmark[lmPose.RIGHT_SHOULDER].x * frame_width),
            int(lm.landmark[lmPose.RIGHT_SHOULDER].y * frame_height),
        )

        # Both ears
        landmarks["l_ear"] = (
            int(lm.landmark[lmPose.LEFT_EAR].x * frame_width),
            int(lm.landmark[lmPose.LEFT_EAR].y * frame_height),
        )

        landmarks["r_ear"] = (
            int(lm.landmark[lmPose.RIGHT_EAR].x * frame_width),
            int(lm.landmark[lmPose.RIGHT_EAR].y * frame_height),
        )

        # Left hip
        landmarks["l_hip"] = (
            int(lm.landmark[lmPose.LEFT_HIP].x * frame_width),
            int(lm.landmark[lmPose.LEFT_HIP].y * frame_height),
        )

        # Right hip
        landmarks["r_hip"] = (
            int(lm.landmark[lmPose.RIGHT_HIP].x * frame_width),
            int(lm.landmark[lmPose.RIGHT_HIP].y * frame_height),
        )

        # Calculate visibility scores
        l_ear_vis = (
            lm.landmark[lmPose.LEFT_EAR].visibility
            if hasattr(lm.landmark[lmPose.LEFT_EAR], "visibility") else 0
        )

        r_ear_vis = (
            lm.landmark[lmPose.RIGHT_EAR].visibility
            if hasattr(lm.landmark[lmPose.RIGHT_EAR], "visibility") else 0
        )

        l_hip_vis = (
            lm.landmark[lmPose.LEFT_HIP].visibility
            if hasattr(lm.landmark[lmPose.LEFT_HIP], "visibility") else 0
        )

        r_hip_vis = (
            lm.landmark[lmPose.RIGHT_HIP].visibility
            if hasattr(lm.landmark[lmPose.RIGHT_HIP], "visibility") else 0
        )

        l_shoulder_vis = (
            lm.landmark[lmPose.LEFT_SHOULDER].visibility
            if hasattr(lm.landmark[lmPose.LEFT_SHOULDER], "visibility")
            else 0
        )

        r_shoulder_vis = (
            lm.landmark[lmPose.RIGHT_SHOULDER].visibility
            if hasattr(lm.landmark[lmPose.RIGHT_SHOULDER], "visibility")
            else 0
        )

        # Add visibility information
        landmarks["primary_ear"] = "left" if l_ear_vis >= r_ear_vis else "right"
        landmarks["l_ear_visibility"] = l_ear_vis
        landmarks["r_ear_visibility"] = r_ear_vis
        landmarks["l_hip_visibility"] = l_hip_vis
        landmarks["r_hip_visibility"] = r_hip_vis
        landmarks["l_shoulder_visibility"] = l_shoulder_vis
        landmarks["r_shoulder_visibility"] = r_shoulder_vis

        return landmarks
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return {}


def analyze_image(image_path, pose, mp_pose, analyzer, sensitivity=-1, visualize=False):
    """
    Analyze posture in a single image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Get image dimensions
    h, w = image.shape[:2]

    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = pose.process(rgb_image)

    # Check if pose was detected
    if not results.pose_landmarks:
        print(f"No pose detected in {image_path}")
        return None

    # Extract landmarks
    landmarks = extract_landmarks(results.pose_landmarks, w, h, mp_pose)

    # Check if landmarks were extracted successfully
    if not landmarks:
        print(f"Failed to extract landmarks from {image_path}")
        return None

    # Analyze posture
    analysis_results = analyzer.analyze_posture(landmarks, sensitivity)

    # Add image path to results
    analysis_results["image_path"] = image_path

    # Visualize if requested
    if visualize:
        # Draw landmarks and posture lines
        color = (0, 255, 0) if analysis_results["good_posture"] else (0, 0, 255)

        # Draw landmarks
        for point_name, point in landmarks.items():
            if isinstance(point, tuple) and len(point) == 2:
                cv2.circle(image, point, 5, color, -1)

        # Draw lines for posture analysis
        if "l_shoulder" in landmarks and "l_ear" in landmarks:
            cv2.line(image, landmarks["l_shoulder"], landmarks["l_ear"], color, 2)

        if "l_shoulder" in landmarks and "l_hip" in landmarks:
            cv2.line(image, landmarks["l_shoulder"], landmarks["l_hip"], color, 2)

        # Add text with angles
        neck_angle = analysis_results.get("neck_angle", "N/A")
        torso_angle = analysis_results.get("torso_angle", "N/A")

        cv2.putText(image, f"Neck: {neck_angle}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Torso: {torso_angle}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Posture: {'Good' if analysis_results['good_posture'] else 'Bad'}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Save the visualized image
        output_dir = os.path.join(os.path.dirname(image_path), "analyzed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)

    return analysis_results


def main():
    parser = argparse.ArgumentParser(description="Analyze posture in images and create a CSV file")
    parser.add_argument("--input_dir", default="./images/", help="Directory containing posture images")
    parser.add_argument("--output_csv", default="output.csv", help="Output CSV file path")
    parser.add_argument("--sensitivity", type=int, default=75, help="Sensitivity threshold (0-100)")
    parser.add_argument("--visualize", action="store_true", help="Visualize analysis results")
    args = parser.parse_args()

    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Initialize posture analyzer
    analyzer = PostureAnalyzer()

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images to analyze")

    # Analyze each image
    results = []
    for image_path in tqdm(image_files, desc="Analyzing images"):
        analysis_result = analyze_image(
            image_path,
            pose,
            mp_pose,
            analyzer,
            sensitivity=args.sensitivity,
            visualize=args.visualize
        )

        if analysis_result:
            results.append(analysis_result)

    # Convert results to DataFrame
    if results:
        # Extract relevant features for the CSV
        csv_data = []
        for result in results:
            csv_data.append({
                'neck_angle': result.get('neck_angle'),
                'torso_angle': result.get('torso_angle'),
                'shoulders_offset': result.get('shoulders_offset'),
                'relative_neck_angle': result.get('relative_neck_angle'),
                'good_posture': result.get('good_posture'),
            })

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(args.output_csv, index=False)
        print(f"Analysis complete. Results saved to {args.output_csv}")

        # Print summary statistics
        good_posture_count = sum(1 for r in results if r['good_posture'])
        bad_posture_count = len(results) - good_posture_count
        print(f"Summary: {good_posture_count} images with good posture, {bad_posture_count} with bad posture")
    else:
        print("No valid results to save.")


if __name__ == "__main__":
    main()
