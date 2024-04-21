import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from langsam_interface.srv import BoundingBoxPrediction
from cv_bridge import CvBridge

from PIL import Image as PILImage
import torch
from langsam_vision.lang_sam import LangSAM
# Import the BoundingBoxes message definition
from langsam_interface.msg import BoundingBoxes

class LangSAMService(Node):
    def __init__(self):
        super().__init__('langsam_service')
        self.subscriber = self.create_subscription(Image, '/color/image_raw', self.image_callback, 10)
        self.service = self.create_service(BoundingBoxPrediction, 'get_bounding_box', self.handle_prediction_request)
        self.cv_bridge = CvBridge()
        self.model = LangSAM()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        self.received = False
        self.get_logger().info('LangSAM Node Initialized with CUDA: ' + str(torch.cuda.is_available()))

    def image_callback(self, img_msg):
        self.pil_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        self.pil_image = PILImage.fromarray(self.pil_image)
        self.pil_image = self.pil_image.convert('RGB')
        if not self.received:
            self.get_logger().info('Image received and processed')
            self.received = True

    def handle_prediction_request(self, request, response):
        while not self.received:
            self.get_logger().info('Waiting for RGB image...')
            rclpy.spin_once(self, timeout_sec=0.1)  # Briefly yield control to allow other callbacks to run
        text_prompt = request.prompt # Get the text prompt from the request
        masks, boxes, phrases, logits = self.model.predict(self.pil_image, text_prompt)
        
        # Prepare bounding box message
        response_boxes = []
        for box in boxes:
            bbox = BoundingBoxes()
            bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax = box
            response_boxes.append(bbox)
        
        response.boxes = response_boxes
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LangSAMService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
