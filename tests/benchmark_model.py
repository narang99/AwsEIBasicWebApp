import os
import io
import logging
import torch, torcheia
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
        model.eval()
        torch._C._jit_set_profiling_executor(False)
        self.model = torcheia.jit.attach_eia(model, 0)
        self.initialized = True

    def inference(self, batch):
        with torch.no_grad():
          with torch.jit.optimized_execution(True):
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

class SysProp:
    def get(self, name):
        assert name == 'model_dir'
        return '/home/ubuntu/qure/awsei/resnet/'
    
class Context:
    system_properties = SysProp()

_service = BeeAntHandler()
_service.initialize(Context())


def create_data(times):
    path = './ant.jpg'
    res = []
    for _ in range(times):
        with open(path, 'rb') as f:
            res.append(f.read())
    return res

tfs = T.Compose([
    T.Resize([256,256]),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
def pil_read(data):
    return torch.stack(list(map(lambda d: tfs(Image.open(io.BytesIO(d))), data)))

def do_f(tns):
    return _service.postprocess(_service.inference(tns))

def warmup(data):
    tns = pil_read(data)
    _service.inference(tns)
    _service.inference(tns)
    _service.inference(tns)

import timeit
data = create_data(64)
warmup(data)
def once():
    tns = pil_read(data)
    do_f(tns)

print(timeit.timeit('once()', number=10, globals=locals()) / 10)


