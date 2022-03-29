

# Usage

From the root directory run
```
docker-compose up
```
This will download two images, one for the backend server running MMS on CPU (no GPU support for default installation) and an image for the frontend server.  
Wait around 10-15 seconds for the backend server to initialize, then go to URL http://localhost:8000/index.html, upload your image and submit. The classifier works on bees and ant images, there are two in the directory `./tests/` in case you want to upload them.  

## Using AWS EC2 EI
Launch an EC2 instance which is configured to use Elastic Inference. [This](https://docs.aws.amazon.com/elastic-inference/latest/developerguide/setting-up-ei.html) is the detailed link. The quick way to do it is to go [here](https://aws.amazon.com/blogs/machine-learning/launch-ei-accelerators-in-minutes-with-the-amazon-elastic-inference-setup-tool-for-ec2/), download the script and run it locally. It will create an instance with EI. (**NOTE:** Only tested in Ubuntu AMI, should work in AL2 AMI but not tested).  

Now follow these steps:
1. **IMPORTANT STEP:** Make sure you can access the instance using HTTP at port 8000, you will need to configure the security group of the EC2 instance and add an entry for `Custom TCP, port 8000`.  
2. SSH into the instance and copy the file `./awsei-docker-compose.yaml` to some directory in your instance as `docker-compose.yaml`.
3. Run `docker-compose up`
4. **NOTE:** if you have cloned the project in the instance, go to root project directory and run `docker-compose -f awsei-docker-compose.yaml`

In your browser, you can access the endpoint using `http://<ec2-public-ip>:8000/index.html`. You will get the IP from AWS console.  

## Building images
Building the images from scratch is a bit more involved for AWS EI. If you do not want to test the build functionality, you can skip this safely.  

### Locally (only CPU inference supported)

Go to the `./build` directory. Run `docker-compose up` from this directory. This will build the images in directory `./cpu`(backend MMS server) and `./local`(frontend server). Go to http://localhost:8000/index.html.  

## Building the AWS EI image
1. Locally building AWS EI image requires you to login using the docker CLI. This is because it uses a base AWS image which requires login. To do this, run the command `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com` in your instance.  For more information, see [this](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)
2. SSH into your instance.
3. Copy this project there (or clone it from git)
4. Go to `build` directory
5. Run `docker-compose -f awsei-docker-compose.yaml up` from the `build` directory.
This will build the AWS EI backend MMS server and the frontend server. In your instance you can access http://localhost:8000/index.html. If your instance's security group allows HTTP at port 8000, you can access the link from your browser using `http://<public-ec2-ip>:8000/index.html`   

# Architecture
The frontend website is really simple as I have not worked with HTML/CSS for a long time and decided to concentrate more on making the backend performant. React seemed overkill for a simple webapp like this.

The backend uses [AWS Multi Model Server](https://github.com/awslabs/multi-model-server) as a model server and PyTorch and torchvision as the DL frameworks. 

A lot of design choices I made were to save expenses (using AWS EC2 EI). 
- The overall design is pretty much what you would expect from a model inference stack (a model server managing the workers, each worker configured to work with our DL framework), another server handling the frontend traffic (this server would also handle other tasks like authentication, any tasks which are not DL specific). Generally people put all of this behind a proxy like Nginx, but since I didn't really need the other compute stuff, I decided to not use a reverse proxy (it makes it more complex anyways).
- There are conflicting views on whether you should use a single model server for multiple models or a single server per model. I feel the single server per model makes it easier to manage. To use this architecture however, extensive testing would be needed because:
  - We do not know how GPU work scheduling would be affected by multiple servers on the node. Having a central server manage the GPUs sounds much better. 

## Requirements considered for backend
- Request batching: This was really important because on GPU machines, if there is no request batching the GPU is highly underutilized
- Working with AWS EC2 Elastic Inference: I don't have a GPU in my machine and the other options on AWS were really costly. AWS was also not giving me dedicated g4dn GPU instances (the cheaper ones). Therefore AWS EC2 Elastic Inference was a cheap and doable option for a simple toy project.

### AWS Elastic Inference (AWS EI)
This is a very cheap GPU offering by AWS which does not provide dedicated GPU to the host, it actually uses network to connect to a pool of GPUs and provides on demand compute. Due to these reasons I ended up using it     
Using AWS EI however, has many annoying problems:
1. All the AMIs and images only support Python 3.6
2. You need a special AWS version of PyTorch integrated with Elastic inference for this to work, and the latest PyTorch version it provides is 1.5.1
3. nvidia-smi does not work, I wanted to get stats for GPU utilization to see if performance tuning could be done but was unable to do so.

All in all, the whole old PyTorch version is a very big problem and mostly would count EI out in prod systems. I still went ahead considering the cost benefits

## Option 1: Flask
There are many variants of using Flask as a server, most notably together with gunicorn as the worker manager. The major limitation of this approach is that there is no inbuilt request batching. [This library](https://github.com/ShannonAI/service-streamer) does some work to overcome the batching issue and actually ends up providing a batch to the model, but the problem with this was:
- The model does get a list of images as inputs which in turn returns a list of outputs, but there was no trivial way to actually find out which request an output image belonged to. Even the example really just returns the whole list of outputs as JSON response for all requests.
Due to the above issues, I decided not to use this stack, I was also scared GIL might end up annoying us later on, but thats just a guess.

## Option 2: torchserve
[TorchServe](https://github.com/pytorch/serve) is a cool model serving framework which takes care of batching for us automatically. All you have to do is provide a JIT model and the server would start serving it with minimal configuration.  
If I have a dedicated CUDA instance, I would strongly consider using torchserve as even the PyTorch team is promoting it.  
Using torchserve on AWS Elastic Inference however, was not possible because:
- The specific version of AWS PyTorch which works with EI only supports Python 3.6 and torchserve requires Python > 3.8    

## Option 3: Multi-model-server (MMS)
While finding an alternative for torchserve, I came across multi-model-server. The project looks like a predecessor of TorchServe (both have similar APIs, although MMS API is clunky, both are based on Netty). Therefore I gave MMS a shot.
The combination of MMS and PyTorch itself is supported nicely by AWS, they provide a base docker image with both of these setup for EI.

# Working with MMS
To serve a model using MMS, you need to follow these steps:
1. Create a handler for this model, this handler is called by MMS for batch of inputs
2. Create a model archive
3. Ask MMS to serve this model archive  

The handler includes data transforms for input (PIL to tensor conversion), loads a JIT model and passes the transformed input to the model. We need to do model transforms because PyTorch 1.5.1 does not support JIT for all the `torchvision` transforms. Here we see another major convenience for using `torchvision`. A typical workflow can be the scientist providing us with a JIT model containing necessary transforms, all our handlers have to do is pass a tensor of the image like this `ToTensor()(PIL.Image())`. 

# Project Structure
```
awsei
  Dockerfile
  modelstore
  resnet
    batch.py
    resnet18.pt
cpu
  Dockerfile
  modelstore
  resnet
    batch.py
    resnet18.pt
local
  template
    index.html
  setup.sh
  Dockerfile
train
  model.ipynb
  model.py
  state_dict.pt
tests
  benchmark_model.py
  ant.jpg
  bee.jpg
```

- Directories `cpu` and `awsei` are different versions of source code for the backend. Building `cpu` would create an image with MMS serving a bee/ant model in a CPU environment. `awsei` is specific image for serving from an AWS EC2 instance configured to use EI.
- `train` is just a throwaway directory I created. It just contains the model code (`model.ipynb` for training copied from internet along with the dataset), `model.py` code which contains the model definition and running it generates the file `resnet18.pt`, the JIT scripted model's dump that we need to serve. This folder is just kept for reference.
- `local` is a very basic frontend which just calls the backend with images. Normally I prefer working with React, but I did not have time to brush up my skills and do it. In any case just for form submission it is overkill considering I'll have to deal with npm images and all. Since I wanted to concentrate on testing the performance, I skipped this part due to less time (and copied the HTML/CSS/JS online from various places)  

The main file of interest is `batch.py`, the MMS repository has examples on how to create handlers for different types of models. The handler file should have a single `handle` method which works as an entrypoint for MMS, it calls this method for inference. In this file, we initialize the model, do data transforms in `preprocess`, run the model on the input in `inference` and return the output after `postprocess`.

You can look at the `Dockerfiles` to see how model archives are created and fed to MMS server.

# Performance optimizations

The GPU I used for EI was the cheapest option which provided a throughput of 1 TFLOPs and 2 GB memory. The instance had 2 CPU vcores.  

## On model optimization
There are many techniques to increase the efficiency of models like pruning, half precision training, mixed precision training, quantization, etc. This would however, have required more time. Many of these techniques are also used during training the model and so can be sometimes out of scope for data engineers.  
Plus I would need time to brush up on my PyTorch skills and since it was limited, I skipped this part.

## Flask
The performance was really bad in CPU (response times went upto 20 seconds when I had a load of just 1 request per second. Each request took approximately 1 second on my MAC but still Flask was not able to keep up.)
I didn't move forward with GPU because batching was not supported

## MMS Pytorch with EI
The first benefit we see immediately is with batching. We can nsustain a load of 70 RPS (with some tuning) without big hits to the latency. Please keep in mind that AWS EI is a shared resource and therefore it is common to see variable latency and throughput because of tenants sharing the same hardware.  

The first time I load tested it, I bombarded it with a high throughput maxing out the CPU. I ended up getting connection resets due to this. There is one free optimization we can do for Intel x86_64 processors (majority of these in AWS EI).  
To help with the load on CPU, I uninstalled the default Pillow library and downloaded pillow-simd built with libjpeg-turbo. This optimization is only possible in Intel x86_64 CPUs, which are very common AWS EC2 instances and therefore was worth a shot. I stopped getting the errors after installing it. Not much latency difference was found here, although CPU utilization decreased. Since its a free optimization, I went ahead and did it.  

The model sustains a load of 70-100 rps. Lets do some back of the paper calculations to see how much we can expect.  
```
Resnet18 batch size 1: 3.6 * 10^9 f32 operations  
GPU limit: 1 TFLOP, that is 10^12 f32 operations
Number of Resnet18 applications GPU can do: 277 per second
```

So ideally, we should set a goal of sustaining an RPS=277.  
I could not reach this level at all no matter what configurtion I tried. I had no idea whether this was MMS being slow or a GPU limitation.  

**NOTE**: Simplified calculations like these are not very reliable. Specifically 1 TFLOP just says that it can do 10<sup>12</sup> f32 operations, it does not specify whether its an add or a multiply. I don't clearly understand what they mean and didn't go about researching this and decided to run experiments to see what is the maxima we can actually reach.  

### How much time does the model take?
To get to the root of this, we first need to take MMS out of equation and see how much time does the GPU take for different batch sizes. `tests/benchmark_model.py` is a copy of `awsei/resnet/batch.py`. Instead of talking to the server, we benchmark the handler with our own batch sizes.  
This is how it looks:
| batch size | time  | time per tensor (time / batch_size) |
-------------|-------|-------------------------------------|
|     8      | 140ms | 17.5ms |
|    16      | 260ms | 16.25ms|
|    32      | 440ms | 13.75ms|
|    64      | 1.04s |16.25ms|

We can see that the time for computation per tensor decreases as we go from batch size of 8 to 32, but ends up increasing when we go to 64. In my observations with load testing the MMS server, I saw the same result (throughput and latency both suffered when we switched from 32 to 64).  
This shows that in a second, it is only feasible to do 64 requests. Therefore, if MMS does not do very bad, maintaining a latency of 500ms for an rps of 50 should be a breeze.

I did many configurations (it was pretty random at first), I'm only keeping two here which are based on the above observation.  

### Load testing results for MMS
**NOTE:** If you do the tests yourself on EI, make sure you warm up the models. Generally if there are `n` workers, the first `n` requests would take a long time because they are loading the model in GPU.  

Running the MMS server with this command in AWS EC2 instance with EI:
```
# start the server
docker run -it --rm -d -e BATCH_SIZE=64 -e WORKERS=1 -e LINGER_MS=10 -p 8080:8080 -p 8081:8081 model_x86_64:0.0.0

# benchmark
ab -k -l -n 1000 -c 50 -T "image/jpeg" -p ./ant.jpg localhost:8080/predictions/resnet18

Server Software:
Server Hostname:        localhost
Server Port:            8080

Document Path:          /predictions/resnet18
Document Length:        Variable

Concurrency Level:      50
Time taken for tests:   20.625 seconds
Complete requests:      1000
Failed requests:        0
Keep-Alive requests:    1000
Total transferred:      234000 bytes
Total body sent:        7983000
HTML transferred:       3000 bytes
Requests per second:    48.49 [#/sec] (mean)
Time per request:       1031.229 [ms] (mean)
Time per request:       20.625 [ms] (mean, across all concurrent requests)
Transfer rate:          11.08 [Kbytes/sec] received
                        377.99 kb/s sent
                        389.07 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.4      0       2
Processing:   163 1024 239.1    973    1957
Waiting:      163 1024 239.1    973    1956
Total:        163 1024 239.1    973    1957

Percentage of the requests served within a certain time (ms)
  50%    973
  66%   1012
  75%   1027
  80%   1043
  90%   1163
  95%   1945
  98%   1954
  99%   1955
 100%   1957 (longest request)
```
Doesn't look too bad, but its not the goal we wanted, latencies have taken a hit at 50 rps. The 50th percentile of our requests is at 1100 milliseconds already.  

Lets try running with batch_size=32 and workers=2
```
# start the server
docker run -it --rm -d -e BATCH_SIZE=32 -e WORKERS=2 -e LINGER_MS=10 -p 8080:8080 -p 8081:8081 model_x86_64:0.0.0

# benchmark
Concurrency Level:      50
Time taken for tests:   13.015 seconds
Complete requests:      1000
Failed requests:        0
Keep-Alive requests:    1000
Total transferred:      234000 bytes
Total body sent:        7983000
HTML transferred:       3000 bytes
Requests per second:    76.84 [#/sec] (mean)
Time per request:       650.732 [ms] (mean)
Time per request:       13.015 [ms] (mean, across all concurrent requests)
Transfer rate:          17.56 [Kbytes/sec] received
                        599.01 kb/s sent
                        616.57 kb/s total

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        0    0   0.3      0       2
Processing:   349  642 160.9    678     944
Waiting:      349  642 160.9    678     944
Total:        349  642 161.0    678     945

Percentage of the requests served within a certain time (ms)
  50%    678
  66%    742
  75%    765
  80%    779
  90%    852
  95%    878
  98%    895
  99%    922
 100%    945 (longest request)  
```
Not ideal (500ms latency) but sweet. I used `ab` for benchmarking so its hard to maintain an exact throughput, but we see a mean latency of 700ms for 70 RPS. Not bad.  
On doing 3 workers, the latency started to increase. So I stopped. Its hard to assume what might have gone wrong with three workers, maybe its not getting batched enough.

**NOTE:** I had also used the configuration with batch=32 and workers=1 but this didn't work out very well (the performance was comparable to batch=64 and workers=1). 

# Comparison of benchmarks done by AWS MMS team
You can find their benchmarking on the effects of batching [here](https://github.com/awslabs/multi-model-server/blob/master/docs/batch_inference_with_mms.md#performance-benchmarking)   
Summary:
- Compute power is 8 cores of GPU, each core is 7.8 TFLOPs
  - Assume 60 TFLOPs total power, 60 times more than ours
- Resnet152: 11.61G MAC
  - Approximately 6.5 times more operations than Resnet18
- they can therefore approximately handle 9 (60/6.5) times our throughput at our latency levels. Thats about `70*9 = 630` RPS.
- In the article they do 350 RPS. They have not provided the latencies they got for this test.

There is degradation in expected throughput and I cannot know until I try it out myself, it does however give us confidence in our results.

# Challenges
- It was very hard making everything work in EC2 with EI. Due to the old PyTorch version, I had to carefully find the correct versions for other dependencies (torchvision). 
- I made a mistake in not checking that there really was no Python 3.8 version of EI compatible PyTorch, creating a big road block on using torchserve after I had completed the setup on EC2 (I didn't use docker images in the beginning)
- Variable results due to multitenancy of shared GPU (I'm assuming, although for our best case of batch_size=32 and workers=2, the variance was observed for 5% of the requests)

# Conclusion
Considering the simplicity of the app, I considered trying the case of maximising the hardware usage. There was no access to GPU metrics and so this doesn't look possible to determine. Therefore I skipped creating a good looking frontend due to lack of time. 

Optimal parameters for the application:
```
Batch size: 32
Workers: 2
Maximum concurrent connections: 50
RPS attained: 70
Latencies: 50%: 700ms 95%: 1000ms
```
