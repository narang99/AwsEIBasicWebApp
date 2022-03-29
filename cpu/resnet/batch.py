import os
import io
import logging
import torch
from torchvision import transforms as T
from PIL import Image


class BeeAntHandler(object):
    def __init__(self):
        self.model = None
        self.labels = None
        self.initialized = False
        self.tfs = None

    def initialize(self, context):
        self.tfs = T.Compose([
            T.Resize([256,256]),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model = torch.jit.load(os.path.join(model_dir, 'resnet18.pt'), map_location=torch.device('cpu'))
        self.model = model.eval()
        self.initialized = True

    def inference(self, batch):
        with torch.no_grad():
            return self.model.forward(batch)

    def _read_from_request_data(self, data):
        image = data.get("data")
        if image is None:
            image = data.get("body")
        image = Image.open(io.BytesIO(image))
        return self.tfs(image)

    def preprocess(self, request):
        return torch.stack(list(map(self._read_from_request_data, request)))

    def postprocess(self, data):
        _, yhats = data.max(1)
        return list(map(lambda yhat: 'ant' if yhat.item() == 0 else 'bee', yhats))

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = BeeAntHandler()
def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    try:
        return _service.handle(data, context)
    except Exception as e:
        logging.error(e, exc_info=True)
        request_processor = context.request_processor
        request_processor.report_status(500, "Unknown inference error")
        return [str(e)] * _service._batch_size

