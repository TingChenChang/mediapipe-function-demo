import gradio as gr
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ====== Face Detection ======
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5)

def face_detection_process(image):

    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(image)
    
    annotated_image = image.copy()
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
      
    return annotated_image

# ====== Face Mesh ======
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

def face_mesh_process(image):

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(image)
    
    annotated_image = image.copy()
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
    
    return annotated_image

# ====== Holistic ======

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def holistic_process(image):
    
    BG_COLOR = (192, 192, 192)  # gray

    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = holistic.process(image)
    
    annotated_image = image.copy()
    # Draw pose, left and right hands, and face landmarks on the image.
    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_tesselation_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_pose_landmarks_style())
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_hand_landmarks_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.
            get_default_hand_landmarks_style())
        
    return annotated_image


# ====== Selfie Segemetation ======

selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
    model_selection=0)

def selfie_segementation_process(image):

    BG_COLOR = (192, 192, 192)  # gray
    
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(image)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, image, bg_image)
        
    return output_image


# ====== Gradio ======
def infer(source_frame, type_options):
    
    if type_options == '臉部偵測':
        output_frame = face_detection_process(source_frame)
    elif type_options == '臉部網格':
        output_frame = face_mesh_process(source_frame)
    elif type_options == '動作偵測':
        output_frame = holistic_process(source_frame)
    elif type_options == '人像去背':
        output_frame = selfie_segementation_process(source_frame)
    
    return output_frame

demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(
            label="Camera",
            image_mode='RGB',
            source='webcam',
            streaming=True),
        gr.Dropdown(
            choices=[
                "臉部偵測",
                "臉部網格",
                "動作偵測",
                "人像去背"],
            label="Options",
            value="臉部偵測")
    ],
    outputs=gr.Image(label="Output", source='webcam', streaming=True),
    title="Mediapipe Demo",
    live=True
)

if __name__ == "__main__":
    demo.launch(share=False)
